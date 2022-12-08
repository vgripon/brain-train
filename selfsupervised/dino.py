import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from args import args
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter, ImageOps
import numpy as np

DEFAULT_NCROPS = 10
DEFAULT_GLOBAL_CROPS_SCALE = (0.5,1)
DEFAULT_LOCAL_CROPS_SCALE = (0.05, 0.4)
DEFAULT_HEAD_HIDDEN_DIM = 2048
DEFAULT_OUT_DIM = 256
DEFAULT_BOTTELENECK_DIM = 256
DEFAULT_CENTER_MOMENTUM = 0.9
DEFAULT_STUDENT_TEMPERATURE = 0.1
DEFAULT_WARMUP_TEACHER_TEMP = 0.04
DEFAULT_WARMUP_TEACHER_TEMP_EPOCHS = 30 if args.epochs > 30 else int(0.03*args.epochs)
DEFAULT_TEACHER_TEMPERATURE = 0.04
DEFAULT_MOMENTUM_TEACHER = 0.996

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )   

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DINOAugmentation(object):
    def __init__(self,
                 image_size, normalization, local_crops_number=DEFAULT_NCROPS, global_crops_scale=DEFAULT_GLOBAL_CROPS_SCALE, local_crops_scale=DEFAULT_LOCAL_CROPS_SCALE):
    
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],p = 0.3),
            transforms.RandomGrayscale(p=0.2),
        ])
        # first global crop
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            transforms.ToTensor(),
            normalization,
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            transforms.ToTensor(),
            normalization,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(image_size*96/224), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            normalization,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, head_hidden_dim, bottleneck_dim, norm_last_layer=True):
        super(DINOHead, self).__init__()
        layers = [nn.Linear(in_dim, head_hidden_dim), nn.BatchNorm1d(head_hidden_dim)]
        for i in range(2):
            layers.append(nn.Linear(head_hidden_dim, head_hidden_dim))
            layers.append(nn.BatchNorm1d(head_hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(head_hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
class DINO(nn.Module):
    def __init__(self, in_dim, epochs, nSteps, head_hidden_dim=DEFAULT_HEAD_HIDDEN_DIM, out_dim=DEFAULT_OUT_DIM, bottleneck_dim=DEFAULT_BOTTELENECK_DIM, center_momentum=DEFAULT_CENTER_MOMENTUM, student_temperature=DEFAULT_STUDENT_TEMPERATURE, norm_last_layer=True, warmup_teacher_temp=DEFAULT_WARMUP_TEACHER_TEMP, warmup_teacher_temp_epochs=DEFAULT_WARMUP_TEACHER_TEMP_EPOCHS, teacher_temperature=DEFAULT_TEACHER_TEMPERATURE, momentum_teacher=DEFAULT_MOMENTUM_TEACHER, ncrops=DEFAULT_NCROPS+2):
        super(DINO, self).__init__()
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.nSteps = nSteps
        self.momentum_schedule = cosine_scheduler(momentum_teacher, 1, epochs, nSteps)
        self.centering = nn.Parameter(torch.zeros(in_dim))
        self.student_temperature = student_temperature
        self.teacher_temp_schedule = np.concatenate((
                np.linspace(warmup_teacher_temp,teacher_temperature, warmup_teacher_temp_epochs),
                np.ones(epochs - warmup_teacher_temp_epochs) * teacher_temperature
            ))
        self.register_buffer("center", torch.zeros(1, out_dim)) # won't be updated by optimizer, not returned by model.parameters()
        self.teacher_head = DINOHead(in_dim, out_dim, head_hidden_dim, bottleneck_dim)
        self.student_head = DINOHead(in_dim, out_dim, head_hidden_dim, bottleneck_dim, norm_last_layer=norm_last_layer)

    @torch.no_grad()
    def update_teacher(self, student, teacher, epoch, batchIdx):
        # EMA update for the teacher
        m = self.momentum_schedule[self.nSteps*epoch+batchIdx]
        for param_t, param_s in zip(teacher.parameters(), student.parameters()):
            param_t.data.mul_(m).add_((1 - m)*param_s.detach().data)
        for param_t, param_s in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            param_t.data.mul_(m).add_((1 - m)*param_s.detach().data)

    def forward_multicrops(self, backbone, head, x):
        # forward the backbone on different crops if image size is different
        # convert to list
        if not isinstance(x, list):
            x = [x]
        output = torch.empty(0).to(x[0].device)
        for crop in x:
            _out = backbone(crop)
            if isinstance(_out, tuple): # in case of tuple output
                _out = _out[0]
            _out = head(_out)
            output = torch.cat((output, _out), dim=0)
        return output
        # idx_crops = torch.cumsum(torch.unique_consecutive(
        #     torch.tensor([inp.shape[-1] for inp in x]),
        #     return_counts=True,
        # )[1], 0)
        # start_idx, output = 0, torch.empty(0).to(x[0].device)
        # for end_idx in idx_crops: # split to multiple batches if input size is different for different crops
        #     _out = backbone(torch.cat(x[start_idx: end_idx]))
        #     if isinstance(_out, tuple): # in case of tuple output
        #         _out = _out[0]
        #     # accumulate outputs
        #     output = torch.cat((output, _out))
        #     start_idx = end_idx
        # # Run the head forward on the concatenated features.
        # return head(output)

    def forward(self, student, teacher, dataStep, target, epoch):
        teacher_output = self.forward_multicrops(teacher, self.teacher_head, dataStep[:2]) # only the 2 global views pass through the teacher
        student_output = self.forward_multicrops(student, self.student_head, dataStep)
        student_out = student_output / self.student_temperature
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp_schedule[epoch], dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output) 

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)