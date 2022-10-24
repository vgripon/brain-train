import torch 
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head) -> None:
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads   
        self.heads = heads
        self.scale = dim_head ** -0.5 # 1/sqrt(dim_head)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False) # One linear for all Q, K, V for all heads
        self.softmax = nn.Softmax(dim=-1) # Softmax over num_patches for each head separately
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1) # Split Q, K, V for all heads, (batch, num_patches, dim) -> 3x(batch, num_patches, dim_head*heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # Rearrange to (batch, heads, num_patches, dim_head)
        prod = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale # (batch, heads, num_patches, num_patches)
        prod = self.softmax(prod)
        out = torch.einsum('b h n m, b h m d -> b h n d', prod, v) # (batch, heads, num_patches, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads) # (batch, num_patches, dim_head*heads)
        return self.to_out(out) #(batch, num_patches, dim)

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = Attention(dim, heads, dim_head)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class Transformer(nn.Module):
    """
    Transformer Architecture    
    """
    def __init__(self,  dim, depth, heads, dim_head, mlp_dim,) -> None:
        super(Transformer, self).__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(TransformerLayer(dim, heads, dim_head, mlp_dim))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x) -> torch.Tensor:
        return self.layers(x)

class ViT(nn.Module):
    """
    Visual Transformer
    """
    def __init__(self,
                image_size,
                patch_size,
                channels,
                dim,
                depth,
                heads,
                mlp_dim,
                dim_head=64, 
                pool=True) -> None:
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim =  patch_height * patch_width * channels

        self.patch_to_embedding = nn.Sequential(
            # Rearrange from (batch, channels, H, W) to (batch, num_patches, patch_dim) num_patches = channels * (H//patch_height) * (W//patch_width)
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width), # Let Rearrange figure out h and w which is num_patches
            nn.Linear(patch_dim, dim))
            
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # Trainable parameter Add 1 for cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # Trainable parameter for the class token, refers to the task at hand used for training the transformer.
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = pool
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, norm=True, mixup=None, lbda = None, perm = None) -> torch.Tensor:
        
        x = self.patch_to_embedding(x)# Project to embedding space (batch, num_patches, dim)
        b, n = x.shape[:2]
        cls_tokens = self.cls_token.expand(b, -1, -1) # Expand cls_token to (batch, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1) # Concat cls_token to (batch, num_patches+1, dim)
        x += self.pos_embedding[:, :(n+1)] # Add positional embedding, make sure to add only num_patches+1 embeddings in case of variable image size
        x = self.transformer(x) # Pass through transformer
        features = x.mean(dim=1) if self.pool else x[:,0]# Average pooling for features
        if norm: features = self.norm(features)
        return features
