import torch
from torch import nn
from typing import List

NUM_CHANNELS = 3

# no point in subclassing torchvision ViT. Too many structural changes needed
class Encoder(nn.Module):
    # default arg values are configuration for ViT base w/ 16 patch size. pe_max_width and pe_max_height are the 
    # max dimensions, in patches, for 2d pes this model will support without interpolation
    def __init__(self, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, pe_max_width=32, pe_max_height=96):
        super().__init__()
        self.patch_size = patch_size
        self.pe_max_width = pe_max_width
        self.pe_max_height = pe_max_height
        self.hidden_dim = hidden_dim
        self.pos_embedding = nn.Parameter(
            torch.zeros(self.pe_max_width, self.pe_max_height, self.hidden_dim)
        ) # overwrite positional embeddings to 2d absolute embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.1)

        # assume these images all are RGB (3 channels)
        self.projection = nn.Linear(in_features=(NUM_CHANNELS * self.patch_size ** 2), out_features=self.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, activation="gelu", batch_first=True),
            num_layers=num_layers, 
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6) # copied from ViT source code
        )

    def batchify(self, x: List[torch.Tensor]):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        seq_lens = []
        pos_embed_slices = []
        # modify x in place to contain its tensors patchified. Also keep track of created sequence lengths and 
        # positional embeddings to add later
        for i, t in enumerate(x):
            h_p = t.shape[-2] // self.patch_size
            w_p = t.shape[-1] // self.patch_size
            if h_p > self.pe_max_height or w_p > self.pe_max_width:
                raise ValueError(f"{h_p} x {w_p} image is too large for max positional embedding grid of shape {self.pe_max_height} x {self.pe_max_width}")

            t = unfold(t.unsqueeze(0)) # (C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
            seq_lens.append(t.shape[-1]) 
            pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim) # (h_p x w_p x E) -> ((h_p x w_p) x E) = (L x E)
            pos_embed_slices.append(pos_embed_slice)
            x[i] = t.squeeze(0).transpose(0, 1) # (1 x (CP^2) x L) -> (L x (CP^2)) to prepare for projection

        # project tensors to embedding dimension
        nested_batch = torch.nested.nested_tensor(x)
        padded_batch = nested_batch.to_padded_tensor(padding=0.0) # (B x L_m x (CP^2)) where L_m is max sequence length in batch
        embeddings = self.projection(padded_batch) # (B x L_m x E) where E is hidden dimension

        # add positional embeddings
        nested_pos_embeds = torch.nested.nested_tensor(pos_embed_slices)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0) # (B x L_m x E)
        embeddings = embeddings + padded_pos_embeds

        # use recorded sequence lenghts to create padding mask for attention
        arange = torch.arange(end=embeddings.shape[1]).unsqueeze(0) # (1, L_m)
        seq_lens = torch.tensor(seq_lens).unsqueeze(1) # (B, 1)
        src_key_padding_mask = arange >= seq_lens
        return embeddings, src_key_padding_mask # nested tensors not supported by attention during training

        # un-pad back into nested tensor
        # nested_batch = torch.nested.narrow(embeddings, dim=1, start=0, length=torch.tensor(seq_lens), layout=torch.jagged) # (B x j1 x E)
        # return nested_batch.contiguous() # force metadata like offsets to be consistent with jaggedness

    def forward(self, x: List[torch.Tensor]):
        x, src_key_padding_mask = self.batchify(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask # return mask for later use in loss calculation

# class PreTrainDecoder(VisionTransformer):
# class LMXDecoder(VisionTransformer):