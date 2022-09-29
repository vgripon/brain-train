### util functions

import time
import torch
import numpy
import scipy.stats as st
from args import args
import torch.nn.functional as F
import torch
from torchvision.utils import _log_api_usage_once
from collections.abc import Sequence
import warnings
from absl import logging
from PIL import Image
from PIL import ImageOps
import io
import os
import itertools
import sys
import json
import collections

lastDisplay = time.time()
def display(string, end = '\n', force = False):
    global lastDisplay
    if time.time() - lastDisplay > 0.1 or force:
        lastDisplay = time.time()
        print(string, end=end)

def timeToStr(time):
    hours = int(time) // 3600
    minutes = (int(time) % 3600) // 60
    seconds = int(time) % 60
    return "{:d}h{:02d}m{:02d}s".format(hours, minutes, seconds)

def confInterval(scores):
    if scores.shape[0] == 1:
        low, up = -1., -1.
    elif scores.shape[0] < 30:
        low, up = st.t.interval(0.95, df = scores.shape[0] - 1, loc = scores.mean(), scale = st.sem(scores.numpy()))
    else:
        low, up = st.norm.interval(0.95, loc = scores.mean(), scale = st.sem(scores.numpy()))
    return low, up

def createCSV(trainSet, validationSet, testSet):
    if args.csv != "":
        f = open(args.csv, "w")
        text = "epochs, "
        for datasetType in [trainSet, validationSet, testSet]:
            for dataset in datasetType:
                text += dataset["name"] + " loss, " + dataset["name"] + " accuracy, "
        f.write(text + "\n")
        f.close()

def updateCSV(stats, epoch = -1):
    if args.csv != "":
        f = open(args.csv, "a")
        text = ""
        if epoch >= 0:
            text += "\n" + str(epoch) + ", "
        for i in range(stats.shape[0]):
            text += str(stats[i,0].item()) + ", " + str(stats[i,1].item()) + ", "
        f.write(text)
        f.close()


class Resize_with_corners(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` modes.
            This can help making the output for PIL images and tensors closer.
    """

    def __init__(self, size, mode='bilinear', antialias=None, align_corners=None):
        super().__init__()
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.align_corners=align_corners
        self.mode = mode
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        #return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        scl_factor = None
        return F.interpolate(img.unsqueeze(0), size=self.size, scale_factor=scl_factor, mode = self.mode, align_corners = self.align_corners, recompute_scale_factor=None, antialias = self.antialias)[0]


    def __repr__(self) -> str:
        detail = f"(size={self.size}, mode={self.mode}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"



def write_example(img, class_label, i, path):
    if not os.path.isdir(os.path.join(path,str(class_label))):
        os.mkdir(os.path.join(path,str(class_label)))
    img.save(os.path.join(path,str(class_label),'img'+str(i)+'.jpg'))

def write_from_image_files(class_files,
                                    class_label,
                                    invert_img=False,
                                    bboxes=None,
                                    output_format='JPEG',
                                    skip_on_error=False, new_path= 'set_new_path'):
  """Create and write a tf.record file for the images corresponding to a class.
  Args:
    class_files: the list of paths to images of class class_label.
    class_label: the label of the class that a record is being made for.
    invert_img: change black pixels to white ones and vice versa. Used for
      Omniglot for example to change the black-background-white-digit images
      into more conventional-looking white-background-black-digit ones.
    bboxes: list of bounding boxes, one for each filename passed as input. If
      provided, images are cropped to those bounding box values.
    output_format: a string representing a PIL.Image encoding type: how the
      image data is encoded inside the tf.record. This needs to be consistent
      with the record_decoder of the DataProvider that will read the file.
    skip_on_error: whether to skip an image if there is an issue in reading it.
      The default it to crash and report the original exception.
  Returns:
    The number of images written into the records file.
  """

  def load_and_process_image(path, bbox=None):
    """Process the image living at path if necessary.
    If the image does not need any processing (inverting, converting to RGB
    for instance), and is in the desired output_format, then the original
    byte representation is returned.
    If that is not the case, the resulting image is encoded to output_format.
    Args:
      path: the path to an image file (e.g. a .png file).
      bbox: bounding box to crop the image to.
    Returns:
      A bytes representation of the encoded image.
    """
    try:
        img = Image.open(path)
    except:
      logging.warn('Failed to open image: %s', path)
      raise

    img_needs_encoding = False

    if img.format != output_format:
      img_needs_encoding = True
    if img.mode != 'RGB':
      img = img.convert('RGB')
      img_needs_encoding = True
    if bbox is not None:
      img = img.crop(bbox)
      img_needs_encoding = True
    if invert_img:
      img = ImageOps.invert(img)
      img_needs_encoding = True

    if img_needs_encoding:
      # Convert the image into output_format
      #buf = io.BytesIO()
      #img.save(buf, format=output_format)
      #buf.seek(0)
      #image_bytes = buf.getvalue()
      pass
    return img
  written_images_count = 0

  for i, path in enumerate(class_files):
    bbox = bboxes[i] if bboxes is not None else None

    img = load_and_process_image(path, bbox)
    

      # This gets executed only if no Exception was raised
    if 'img' in locals():
        write_example(img, class_label, i, path = os.path.join(args.dataset_path, new_path))
        written_images_count += 1

  print( written_images_count)


print(" utils,", end="")
