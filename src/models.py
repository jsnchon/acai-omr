import torch
from torch import nn
import timm.models.vision_transformer as vit

class AdaptivePadPatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, img_tensors):
        embedded_sequences = []
        for img_tensor in img_tensors:
            # patches are (embed_dim, H/patch_size, W/patch_size)
            patches = self.patch_embed(img_tensor.unsqueeze(0)).squeeze(0) # unsqueeze to signal batch size of 1
            # sequence is (sequence_len, embed_dim) where sequence_len = H/patch_size * W/patch_size
            sequence = patches.permute(1, 2, 0).flatten(0, 1)
            embedded_sequences.append(sequence)
        # (B x L x E) where B = batch size, L = max sequence length, E = embedding dim
        batch = torch.nn.utils.rnn.pad_sequence(embedded_sequences, batch_first=True) 
        # (B x L)
        mask = torch.ones_like(batch[..., 0]) 
        for i, seq in enumerate(embedded_sequences):
            mask[i, len(seq):] = 0 # set embedding positions where padding is to 0
        
        return batch, mask

class MaskedAttention(vit.Attention):
    def forward(self, x, mask):
        # all just from timm's source implementation
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1) # convert masks to 2d grids: B x 1 x N x N
        attn = attn.masked_fill(mask == 0, float("-inf")) # mask out attention to and from padding patch embeddings
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0) # if a row is entirely masked, softmax produces nans, so replace those with 0

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x