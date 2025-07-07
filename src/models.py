import torch
from torch import nn

NUM_CHANNELS = 3

# no point in subclassing torchvision ViT. Too many structural changes needed
class Encoder(nn.Module):
    # default arg values are configuration for ViT base w/ 16 patch size. pe_max_width and pe_max_height are the 
    # max dimensions, in patches, for 2d pes this model will support without interpolation
    def __init__(self, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, pe_max_height=32, pe_max_width=96):
        super().__init__()
        self.patch_size = patch_size
        self.pe_max_height = pe_max_height
        self.pe_max_width = pe_max_width
        self.hidden_dim = hidden_dim
        self.pos_embedding = nn.Parameter(
            torch.zeros(self.pe_max_height, self.pe_max_width, self.hidden_dim)
        ) # overwrite positional embeddings to 2d absolute embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.1)

        # assume these images all are RGB (3 channels)
        self.projection = nn.Linear(in_features=(NUM_CHANNELS * self.patch_size ** 2), out_features=self.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, activation="gelu", batch_first=True),
            num_layers=num_layers, 
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6) # eps copied from ViT source code
        )

    def batchify(self, x: list[torch.Tensor]):
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

        # use recorded sequence lengths to create padding mask for attention
        arange = torch.arange(end=embeddings.shape[1]).unsqueeze(0) # (1, L_m)
        seq_lens = torch.tensor(seq_lens).unsqueeze(1) # (B, 1)
        src_key_padding_mask = arange >= seq_lens # (B, L_m)
        return embeddings, src_key_padding_mask # nested tensors not supported by attention during training

        # un-pad back into nested tensor
        # nested_batch = torch.nested.narrow(embeddings, dim=1, start=0, length=torch.tensor(seq_lens), layout=torch.jagged) # (B x j1 x E)
        # return nested_batch.contiguous() # force metadata like offsets to be consistent with jaggedness

    def forward(self, x: list[torch.Tensor]):
        x, src_key_padding_mask = self.batchify(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask # return mask for later use in loss calculation

# identical model architecture to standard encoder superclass so learned state dict can be transferred to it after
# pre-training. This subclass just modifies the forward logic to include MAE logic (patching -> shuffling -> masking -> encoding)
class MAEEncoder(Encoder):
    def __init__(self, mask_ratio, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, pe_max_height=32, pe_max_width=96):
        super().__init__(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, pe_max_height, pe_max_width)
        self.mask_ratio = mask_ratio

    # shuffle and mask patchified sequence (based off approach used here: https://github.com/facebookresearch/mae/blob/main/models_mae.py)
    def mask_sequence(self, t: torch.Tensor, h_p: int, w_p: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        # shuffle patches
        unmasked_seq_len = t.shape[-1]
        len_keep = int(unmasked_seq_len * (1 - self.mask_ratio)) # how many patches to keep in sequence
        noise = torch.rand(unmasked_seq_len, device=t.device)
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)
        ids_keep = ids_shuffle[:len_keep] # (L, ) -> (L_keep, )
        t_masked = t.index_select(dim=-1, index=ids_keep) # use ids_keep to select L_keep 1 x CP^2 patch tensors. t_masked is (1 x CP^2 x L_keep)

        # record which patches in original sequence are masked
        seq_mask = torch.ones(unmasked_seq_len, dtype=torch.int) # (L, ), 0 means patch was kept, 1 means patch was masked from original sequence
        seq_mask[:len_keep] = 0
        seq_mask = seq_mask.index_select(dim=0, index=ids_restore) # match mask to original sequence order (not shuffled sequence order)

        # slice positional embedding to align with original patch sequence, apply same shuffle/chop to align it to kept sequence
        pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim) 
        pos_embed_slice = pos_embed_slice.index_select(dim=0, index=ids_keep) # (L_keep x E), L_keep first since embedding dims will match this order

        return t_masked, pos_embed_slice, len_keep, seq_mask, ids_restore

    def batchify(self, x: list[torch.Tensor]):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        kept_seq_lens = []
        pos_embed_slices = []
        seq_masks = [] # mask tensors for each sequence recording which patches are masked
        restore_tensors = [] # index tensors to later undo shuffling
        # modify x in place to contain its tensors patchified then shuffled/masked
        for i, t in enumerate(x):
            h_p = t.shape[-2] // self.patch_size # record these before unfolding
            w_p = t.shape[-1] // self.patch_size
            if h_p > self.pe_max_height or w_p > self.pe_max_width:
                raise ValueError(f"{h_p} x {w_p} image is too large for max positional embedding grid of shape {self.pe_max_height} x {self.pe_max_width}")

            t = unfold(t) # (C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
            t_masked, pos_embed_slice, len_keep, seq_mask, ids_restore = self.mask_sequence(t, h_p, w_p)
            kept_seq_lens.append(len_keep)
            seq_masks.append(seq_mask)
            restore_tensors.append(ids_restore)
            pos_embed_slices.append(pos_embed_slice)

            x[i] = t_masked.squeeze(0).transpose(0, 1) 

        # project tensors to embedding dimension
        nested_batch = torch.nested.nested_tensor(x)
        padded_batch = nested_batch.to_padded_tensor(padding=0.0) 
        embeddings = self.projection(padded_batch) 

        # add positional embeddings
        nested_pos_embeds = torch.nested.nested_tensor(pos_embed_slices)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0) 
        embeddings = embeddings + padded_pos_embeds

        # use recorded sequence lengths to create padding mask for attention
        arange = torch.arange(end=embeddings.shape[1]).unsqueeze(0) 
        kept_seq_lens = torch.tensor(kept_seq_lens).unsqueeze(1) 
        src_key_padding_mask = arange >= kept_seq_lens

        # create tensors to use later 
        batch_seq_masks = torch.nested.nested_tensor(seq_masks, layout=torch.jagged) # (N x j1)
        batch_ids_restore = torch.nested.nested_tensor(restore_tensors, layout=torch.jagged) # (N x j1)
        return embeddings, src_key_padding_mask, batch_seq_masks, batch_ids_restore

    def forward(self, x: list[torch.Tensor]):
        x, src_key_padding_mask, batch_seq_masks, batch_ids_restore = self.batchify(x)
        # x now only contains the patches that weren't masked so we only encode visible patches
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask, batch_seq_masks, batch_ids_restore

# MAE decoder needs learnable mask token and also to specify decoder dim (since can be narrower)

# class PreTrainDecoder(VisionTransformer):
# class LMXDecoder(VisionTransformer):