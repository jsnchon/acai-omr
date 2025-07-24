import torch
from torch import nn

NUM_CHANNELS = 1 # assume these images all are grayscale

# no point in subclassing torchvision ViT. Too many structural changes needed
class Encoder(nn.Module):
    # default arg values are configuration for ViT base w/ 16 patch size. pe_max_width and pe_max_height are the 
    # max dimensions, in patches, for 2d pes this model will support without interpolation
    def __init__(self, patch_size=16, num_layers=12, hidden_dim=768, num_heads=12, mlp_dim=3072, pe_max_height=32, pe_max_width=96, transformer_dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.pe_max_height = pe_max_height
        self.pe_max_width = pe_max_width
        self.hidden_dim = hidden_dim
        self.pos_embedding = nn.Parameter(
            torch.zeros(self.pe_max_height, self.pe_max_width, self.hidden_dim)
        ) 
        nn.init.trunc_normal_(self.pos_embedding, std=0.1)

        self.projection = nn.Linear(in_features=(NUM_CHANNELS * self.patch_size ** 2), out_features=self.hidden_dim)
        self.encoder_blocks = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=transformer_dropout, activation="gelu", batch_first=True),
            num_layers=num_layers, 
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6) # eps copied from ViT source code
        )

    def batchify(self, x: list[torch.Tensor]):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        seq_lens = []
        pos_embed_slices = []
        patchified_tensors = []
        # patchify tensors to group later. Also keep track of created sequence lengths and 
        # positional embeddings to add later
        for t in x:
            h_p = t.shape[-2] // self.patch_size
            w_p = t.shape[-1] // self.patch_size
            if h_p > self.pe_max_height or w_p > self.pe_max_width:
                raise ValueError(f"{h_p} x {w_p} image is too large for max positional embedding grid of shape {self.pe_max_height} x {self.pe_max_width}")

            t = unfold(t.unsqueeze(0)) # (1 x C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
            seq_lens.append(t.shape[-1]) 
            pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim) # (h_p x w_p x E) -> ((h_p x w_p) x E) = (L x E)
            pos_embed_slices.append(pos_embed_slice)
            patchified_tensors.append(t.squeeze(0).transpose(0, 1)) # (1 x (CP^2) x L) -> (L x (CP^2)) to prepare for projection

        # project tensors to embedding dimension
        nested_batch = torch.nested.as_nested_tensor(patchified_tensors, layout=torch.jagged)
        padded_batch = nested_batch.to_padded_tensor(padding=0.0) # (B x L_m x (CP^2)) where L_m is max sequence length in batch
        embeddings = self.projection(padded_batch) # (B x L_m x E) where E is hidden dimension

        # add positional embeddings
        nested_pos_embeds = torch.nested.as_nested_tensor(pos_embed_slices, layout=torch.jagged)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0) # (B x L_m x E)
        embeddings = embeddings + padded_pos_embeds

        # use recorded sequence lengths to create padding mask for attention
        src_key_padding_mask = self.create_attention_mask(seq_lens, embeddings.shape[1])
        return embeddings, src_key_padding_mask # nested tensors not supported by attention during training

    # takes a list of batch sequence lengths seq_lens where the ith entry is the ith example's sequence length and a max sequence 
    # length in the batch max_len and returns a (B, L_m) src key attention mask indicating what embeddings to ignore according to seq_lens
    def create_attention_mask(self, seq_lens: list[int], max_len: int):
        arange = torch.arange(end=max_len).unsqueeze(0) # (1, L_m)
        seq_lens = torch.tensor(seq_lens).unsqueeze(1) # (B, 1)
        return arange >= seq_lens # (B, L_m)

    # x is a list of (C x H x W) image tensors
    def forward(self, x: list[torch.Tensor]):
        x, src_key_padding_mask = self.batchify(x)
        x = self.encoder_blocks(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask # return mask for later use in loss calculation

# identical model architecture to standard encoder superclass so learned state dict can be transferred to it after
# pre-training. This subclass just modifies the forward logic to include MAE logic (patching -> shuffling -> masking -> encoding)
class MAEEncoder(Encoder):
    def __init__(self, mask_ratio, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, pe_max_height=32, pe_max_width=96):
        super().__init__(patch_size, num_layers, hidden_dim, num_heads, mlp_dim, pe_max_height, pe_max_width, transformer_dropout=0.0)
        self.mask_ratio = mask_ratio

    # shuffle and mask patchified sequence (based off approach used here: https://github.com/facebookresearch/mae/blob/main/models_mae.py)
    def mask_sequence(self, t: torch.Tensor, h_p: int, w_p: int) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
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

        # slice positional embedding to align with original patch sequence, apply same index select to align it to kept sequence
        pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim) 
        pos_embed_slice = pos_embed_slice.index_select(dim=0, index=ids_keep) # (L_keep x E), L_keep dim first since embedding dims will match this order

        return t_masked, pos_embed_slice, unmasked_seq_len, len_keep, seq_mask, ids_restore

    # x is a list of (C x H x W) image tensors
    def batchify(self, x: list[torch.Tensor]):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        unmasked_seq_lens = []
        kept_seq_lens = []
        pos_embed_slices = []
        patchified_dims = [] # list of (h_p, w_p) tuples for each image. Needed later for decoder positional embedding
        seq_masks = [] # mask tensors for each sequence recording which patches are masked
        restore_tensors = [] # index tensors to later undo shuffling
        masked_tensors = [] 
        for t in x:
            h_p = t.shape[-2] // self.patch_size # record these before unfolding
            w_p = t.shape[-1] // self.patch_size
            if h_p > self.pe_max_height or w_p > self.pe_max_width:
                raise ValueError(f"{h_p} x {w_p} image is too large for max positional embedding grid of shape {self.pe_max_height} x {self.pe_max_width}")
            patchified_dims.append((h_p, w_p))

            t = unfold(t.unsqueeze(0)) # (1 x C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
            t_masked, pos_embed_slice, unmasked_seq_len, len_keep, seq_mask, ids_restore = self.mask_sequence(t, h_p, w_p)
            unmasked_seq_lens.append(unmasked_seq_len)

            kept_seq_lens.append(len_keep)
            seq_masks.append(seq_mask)
            restore_tensors.append(ids_restore)
            pos_embed_slices.append(pos_embed_slice)

            masked_tensors.append(t_masked.squeeze(0).transpose(0, 1))

        # project tensors to embedding dimension
        nested_batch = torch.nested.as_nested_tensor(masked_tensors, layout=torch.jagged)
        padded_batch = nested_batch.to_padded_tensor(padding=0.0) 
        embeddings = self.projection(padded_batch) 

        # add positional embeddings
        nested_pos_embeds = torch.nested.as_nested_tensor(pos_embed_slices, layout=torch.jagged)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0) 
        embeddings = embeddings + padded_pos_embeds

        # attention masks differ between encoder and decoder since encoder only operates on visible patches. Decoder
        # needs a mask for the original sequence lengths since will operate on visible and masked patches
        encoder_attention_mask = self.create_attention_mask(kept_seq_lens, embeddings.shape[1])
        decoder_attention_mask = self.create_attention_mask(unmasked_seq_lens, max(unmasked_seq_lens))

        # create tensors to use later 
        batch_seq_masks = torch.nested.as_nested_tensor(seq_masks, layout=torch.jagged) # (N x j1). nested_tensor automatically adds batch dimension
        batch_ids_restore = torch.nested.as_nested_tensor(restore_tensors, layout=torch.jagged) # (N x j1)

        return embeddings, encoder_attention_mask, decoder_attention_mask, kept_seq_lens, unmasked_seq_lens, batch_seq_masks, batch_ids_restore, patchified_dims

    # x is a list of (C x H x W) image tensors
    def forward(self, x: list[torch.Tensor]):
        x, encoder_attention_mask, decoder_attention_mask, kept_seq_lens, unmasked_seq_lens, batch_seq_masks, batch_ids_restore, patchified_dims = self.batchify(x)
        # x now only contains the patches that weren't masked so we only encode visible patches
        x = self.encoder_blocks(x, src_key_padding_mask=encoder_attention_mask)
        return x, decoder_attention_mask, kept_seq_lens, unmasked_seq_lens, batch_seq_masks, batch_ids_restore, patchified_dims

class Decoder(nn.Module):
    # these default args are for the best performing MAE decoder in the paper (excluding dropout which is just PyTorch transformer default)
    def __init__(self, num_layers=8, hidden_dim=512, num_heads=16, mlp_dim=3072, transformer_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_blocks = nn.TransformerEncoder( # nn.TransformerEncoder works fine here since just self-attending to one sequence
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=transformer_dropout, activation="gelu", batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6)
        )

    # for MAE, assume x is already positionally embedded (so mask tokens have positional info). Don't do it here because
    # positional embeddings aren't needed for after pre-training encoders when encoding entire images
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        return self.decoder_blocks(x, src_key_padding_mask=attention_mask)

class MAE(nn.Module):
    def __init__(self, mask_ratio, patch_size, encoder_hidden_dim=768, decoder_hidden_dim=512, pe_max_height=32, pe_max_width=96, encoder_kwargs={}, decoder_kwargs={}):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = MAEEncoder(mask_ratio, self.patch_size, hidden_dim=encoder_hidden_dim, pe_max_height=pe_max_height, pe_max_width=pe_max_width, **encoder_kwargs)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder = Decoder(hidden_dim=self.decoder_hidden_dim, transformer_dropout=0.0, **decoder_kwargs)
        self.decoder_embed = nn.Linear(encoder_hidden_dim, self.decoder_hidden_dim)
        self.decoder_unembed = nn.Linear(self.decoder_hidden_dim, NUM_CHANNELS * patch_size ** 2) # project from decoder embedding space to pixel predictions
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_hidden_dim)
        )
        self.decoder_pos_embedding = nn.Parameter(
            torch.zeros(pe_max_height, pe_max_width, self.decoder_hidden_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.1)
        nn.init.trunc_normal_(self.decoder_pos_embedding, std=0.1)

    # takes latent tensor projected to decoder dimension and for each sequence appends mask tokens and unshuffles
    # returns a padded batch tensor that's also been positionally embedded for the decoder
    def prepare_for_decoder(self, latent: torch.Tensor, kept_seq_lens: list[int], unmasked_seq_lens: list[int], batch_ids_restore: torch.Tensor, patchified_dims: list[tuple[int, int]]):
        append_amounts = (torch.tensor(unmasked_seq_lens) - torch.tensor(kept_seq_lens)).tolist()
        reconstructed_sequences = []
        pos_embed_slices = []
        for i, sequence in enumerate(latent.unbind()):
            # remove padding
            sequence = sequence[:kept_seq_lens[i], :]
            # append the sequence's needed amount of mask tokens
            sequence = torch.cat([sequence.unsqueeze(0), self.mask_token.repeat(1, append_amounts[i], 1)], dim=1)
            # unshuffle
            sequence = sequence.index_select(dim=1, index=batch_ids_restore[i, :])
            reconstructed_sequences.append(sequence.squeeze(0)) # remove batch dim value of 1 since nested tensor infers batch dim
            # slice positional embedding for this input sequence
            h_p, w_p = patchified_dims[i]
            pos_embed_slices.append(self.decoder_pos_embedding[:h_p, :w_p, :].reshape(-1, self.decoder_hidden_dim))

        reconstructed_sequences = torch.nested.as_nested_tensor(reconstructed_sequences, layout=torch.jagged)
        reconstructed_sequences = reconstructed_sequences.to_padded_tensor(padding=0.0)

        # positionally embed decoder inputs
        nested_pos_embeds = torch.nested.as_nested_tensor(pos_embed_slices, layout=torch.jagged)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0)
        return reconstructed_sequences + padded_pos_embeds

    # batch is a list of (input, target) tuples where each input is a C x H x W image tensor and each target is a corresponding C x H x W target image tensor
    def forward(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        x = []
        y = []
        for ex in batch:
            x.append(ex[0])
            y.append(ex[1])

        latent, decoder_attention_mask, kept_seq_lens, unmasked_seq_lens, batch_seq_masks, batch_ids_restore, patchified_dims = self.encoder(x)
        latent = self.decoder_embed(latent) # project to decoder embedding space

        latent = self.prepare_for_decoder(latent, kept_seq_lens, unmasked_seq_lens, batch_ids_restore, patchified_dims)
        decoder_hidden_state = self.decoder(latent, decoder_attention_mask)
        pred = self.decoder_unembed(decoder_hidden_state) # (N x L_m x CP^2)

        loss_mask = torch.logical_and(~decoder_attention_mask, batch_seq_masks) # True = patch is not padding for attention (False in attn mask) and is a mask token
        loss_mask = loss_mask.to_padded_tensor(padding=False) # True = use patch in loss calculation, False = ignore patch
        
        # prepare target images for loss calculation (may be different from inputs) by patchifying, changing shape to (L, CP^2), padding
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        unfolded_targets = [unfold(t.unsqueeze(0)).squeeze(0).transpose(0, 1) for t in y]
        target_nested = torch.nested.as_nested_tensor(unfolded_targets, layout=torch.jagged)
        target_padded = target_nested.to_padded_tensor(padding=0.0)

        return pred, loss_mask, target_padded 

class MAELoss(nn.Module):
    # reconstruction target: pixel values (normalized per patch) of portions that were masked
    def forward(self, pred: torch.Tensor, loss_mask: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (N x L_m x CP^2)
        loss_mask: (N x L_m)
        """

        # normalize target patches
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (N, L), mean loss per patch

        loss = (loss * loss_mask).sum() / loss_mask.sum() # mean loss for desired patches
        return loss