import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from functools import partial
from acai_omr.config import LMX_PAD_TOKEN, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from acai_omr.models.caching import KVCache
import re

NUM_CHANNELS = 1 # assume these images all are grayscale

# no point in subclassing torchvision ViT. Too many structural changes needed
class Encoder(nn.Module):
    # default arg values are configuration for ViT base w/ 16 patch size (except for 0 dropout since freezing pre-trained encoder). 
    # pe_max_width and pe_max_height are the max dimensions, in patches, for 2d pes this model will support without interpolation
    def __init__(self, patch_size, pe_max_height, pe_max_width, num_layers=12, hidden_dim=768, num_heads=12, mlp_dim=3072, transformer_dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.pe_max_height = pe_max_height
        self.pe_max_width = pe_max_width
        self.hidden_dim = hidden_dim
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
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

    def batchify(self, x: tuple[torch.Tensor]):
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

            t = self.unfold(t.unsqueeze(0)) # (1 x C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
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
        src_key_padding_mask = self.create_attention_mask(seq_lens, embeddings.shape[1]).to(embeddings.device)
        return embeddings, src_key_padding_mask # nested tensors not supported by attention during training

    # takes a list of batch sequence lengths seq_lens where the ith entry is the ith example's sequence length and a max sequence 
    # length in the batch max_len and returns a (B, L_m) src key attention mask indicating what embeddings to ignore according to seq_lens
    def create_attention_mask(self, seq_lens: list[int], max_len: int):
        arange = torch.arange(end=max_len).unsqueeze(0) # (1, L_m)
        seq_lens = torch.tensor(seq_lens).unsqueeze(1) # (B, 1)
        return arange >= seq_lens # (B, L_m)

    # x is a list of (C x H x W) image tensors
    def forward(self, x: tuple[torch.Tensor]):
        x, src_key_padding_mask = self.batchify(x)
        x = self.encoder_blocks(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask # return mask for later use in loss calculation and cross-attention masking

    def embed_single_image(self, x):
        h_p = x.shape[1] // self.patch_size
        w_p = x.shape[2] // self.patch_size

        x = self.unfold(x.unsqueeze(0))
        x = x.transpose(dim0=1, dim1=2) # prepare for projection
        x = self.projection(x)
        pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim)
        x = x + pos_embed_slice.unsqueeze(0)
        return x

    # x is a singular (C x H x W) image tensor
    def generate(self, x: torch.Tensor):
        x = self.embed_single_image(x)
        x = self.encoder_blocks(x)
        return x

# identical model architecture to standard encoder superclass so learned state dict can be transferred to it after
# pre-training. This subclass just modifies the forward logic to include MAE logic (patching -> shuffling -> masking -> encoding)
class MAEEncoder(Encoder):
    def __init__(self, mask_ratio, patch_size, pe_max_height, pe_max_width, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
        super().__init__(patch_size, pe_max_height, pe_max_width, num_layers, hidden_dim, num_heads, mlp_dim, transformer_dropout=0.0)
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
        seq_mask = torch.ones(unmasked_seq_len, device=t.device, dtype=torch.int) # (L, ), 0 means patch was kept, 1 means patch was masked from original sequence
        seq_mask[:len_keep] = 0
        seq_mask = seq_mask.index_select(dim=0, index=ids_restore) # match mask to original sequence order (not shuffled sequence order)

        # slice positional embedding to align with original patch sequence, apply same index select to align it to kept sequence
        pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim) 
        pos_embed_slice = pos_embed_slice.index_select(dim=0, index=ids_keep) # (L_keep x E), L_keep dim first since embedding dims will match this order

        return t_masked, pos_embed_slice, unmasked_seq_len, len_keep, seq_mask, ids_restore

    # x is a list of (C x H x W) image tensors
    def batchify(self, x: list[torch.Tensor]):
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

            t = self.unfold(t.unsqueeze(0)) # (1 x C x H x W) -> (1 x (CP^2) x L) where P is patch size, L is sequence length (h_p x w_p)
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
        encoder_attention_mask = self.create_attention_mask(kept_seq_lens, embeddings.shape[1]).to(embeddings.device)
        decoder_attention_mask = self.create_attention_mask(unmasked_seq_lens, max(unmasked_seq_lens)).to(embeddings.device)

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

class MAEDecoder(nn.Module):
    # these default args are for the best performing MAE decoder in the paper 
    def __init__(self, num_layers=8, hidden_dim=512, num_heads=16, mlp_dim=3072, transformer_dropout=0.0):
        super().__init__()
        self.decoder_blocks = nn.TransformerEncoder( # nn.TransformerEncoder works fine here since just self-attending to one sequence
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=transformer_dropout, activation="gelu", batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim, eps=1e-6)
        )

    # for MAE, assume x is already positionally embedded (so mask tokens have positional info). Don't do it here because
    # positional embeddings aren't needed for after pre-training encoders when encoding entire images
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        return self.decoder_blocks(x, src_key_padding_mask=attention_mask)

class MAE(nn.Module):
    def __init__(self, mask_ratio, patch_size, pe_max_height, pe_max_width, encoder_hidden_dim=768, decoder_hidden_dim=512, encoder_kwargs={}, decoder_kwargs={}):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = MAEEncoder(mask_ratio, self.patch_size, pe_max_height, pe_max_width, hidden_dim=encoder_hidden_dim, **encoder_kwargs)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder = MAEDecoder(hidden_dim=self.decoder_hidden_dim, **decoder_kwargs)
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

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

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
    """
    Output
        pred, target_padded: (N x L_m x CP^2)
        loss_mask: (N x L_m)
    """
    def forward(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        x, y = zip(*batch)
        x = list(x)
        y = list(y)

        latent, decoder_attention_mask, kept_seq_lens, unmasked_seq_lens, batch_seq_masks, batch_ids_restore, patchified_dims = self.encoder(x)
        latent = self.decoder_embed(latent) # project to decoder embedding space

        latent = self.prepare_for_decoder(latent, kept_seq_lens, unmasked_seq_lens, batch_ids_restore, patchified_dims)
        decoder_hidden_state = self.decoder(latent, decoder_attention_mask)
        pred = self.decoder_unembed(decoder_hidden_state) # (N x L_m x CP^2)

        loss_mask = torch.logical_and(~decoder_attention_mask, batch_seq_masks) # True = patch is not padding for attention (False in attn mask) and is a mask token
        loss_mask = loss_mask.to_padded_tensor(padding=False) # True = use patch in loss calculation, False = ignore patch
        
        # prepare target images for loss calculation (may be different from inputs) by patchifying, changing shape to (L, CP^2), padding
        unfolded_targets = [self.unfold(t.unsqueeze(0)).squeeze(0).transpose(0, 1) for t in y]
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

class OMREncoder(Encoder):
    def interpolate_pe(self, h_p, w_p):
        # reshape to (1 x E x h_old x w_old) for interpolation
        pos_embedding = self.pos_embedding.permute(2, 0, 1).unsqueeze(0)

        interpolated_pos_embedding = nn.functional.interpolate(
            pos_embedding,
            size=(h_p, w_p),
            mode="bilinear",
            align_corners=False
        ) # (1 x E x h_p x w_p)

        return interpolated_pos_embedding.squeeze(0).permute(1, 2, 0) # (h_p x w_p x E)

    def batchify(self, x: tuple[torch.Tensor]):
        seq_lens = []
        pos_embed_slices = []
        patchified_tensors = []
        for t in x:
            h_p = t.shape[-2] // self.patch_size
            w_p = t.shape[-1] // self.patch_size

            t = self.unfold(t.unsqueeze(0)) 
            seq_lens.append(t.shape[-1]) 

            if h_p > self.pe_max_height or w_p > self.pe_max_width:
                pos_embed_slice = self.interpolate_pe(h_p, w_p).reshape(-1, self.hidden_dim)
            else:
                pos_embed_slice = self.pos_embedding[:h_p, :w_p, :].reshape(-1, self.hidden_dim)
            pos_embed_slices.append(pos_embed_slice)
            patchified_tensors.append(t.squeeze(0).transpose(0, 1)) 

        nested_batch = torch.nested.as_nested_tensor(patchified_tensors, layout=torch.jagged)
        padded_batch = nested_batch.to_padded_tensor(padding=0.0) 
        embeddings = self.projection(padded_batch) 

        nested_pos_embeds = torch.nested.as_nested_tensor(pos_embed_slices, layout=torch.jagged)
        padded_pos_embeds = nested_pos_embeds.to_padded_tensor(padding=0.0)
        embeddings = embeddings + padded_pos_embeds

        # src_key_padding_mask will be used for mask signifying which patches in encoder memory are padding during cross-attention
        src_key_padding_mask = self.create_attention_mask(seq_lens, embeddings.shape[1]).to(embeddings.device)
        return embeddings, src_key_padding_mask 

class FineTuneOMREncoder(OMREncoder):
    def __init__(self, patch_size, pe_max_height, pe_max_width, fine_tune_depth, num_layers=12, hidden_dim=768, num_heads=12, mlp_dim=3072, transformer_dropout=0.05):
        super().__init__(patch_size, pe_max_height, pe_max_width, num_layers, hidden_dim, num_heads, mlp_dim)
        assert fine_tune_depth > 0, "If using FineTuneOMREncoder, fine-tune depth should be at least 1"
        del self.encoder_blocks

        self.fine_tune_depth = fine_tune_depth
        self.num_layers = num_layers
        self.num_frozen_layers = self.num_layers - self.fine_tune_depth

        # store so these can be retrieved to convert this into a regular OMREncoder later
        self.superclass_kwargs = {"num_heads": num_heads, "mlp_dim": mlp_dim, "transformer_dropout": transformer_dropout}

        base_encoder_layer_kwargs = {"d_model": self.hidden_dim, "nhead": num_heads, "dim_feedforward": mlp_dim, "activation": "gelu", "batch_first": True}
        if self.num_frozen_layers == 0: # full fine-tune
            self.frozen_blocks = None
        else:
            self.frozen_blocks = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(dropout=0.0, **base_encoder_layer_kwargs),
                num_layers=self.num_frozen_layers
            )

        self.fine_tune_blocks = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(dropout=transformer_dropout, **base_encoder_layer_kwargs),
            num_layers=self.fine_tune_depth,
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6)
        )

    def forward(self, x: tuple[torch.Tensor]):
        x, src_key_padding_mask = self.batchify(x)
        if self.frozen_blocks:
            x = self.frozen_blocks(x, src_key_padding_mask=src_key_padding_mask)
        x = self.fine_tune_blocks(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask 

    # x is a singular (C x H x W) image tensor
    def generate(self, x: torch.Tensor):
        x = self.embed_single_image(x)
        if self.frozen_blocks:
            x = self.frozen_blocks(x)
        x = self.fine_tune_blocks(x)

        return x

class OMRDecoder(nn.Module):
    def __init__(self, max_lmx_seq_len, lmx_vocab_path, num_layers=10, hidden_dim=1024, num_heads=16, mlp_dim=4096, transformer_dropout=0.1):
        super().__init__()
        self.max_lmx_seq_len = max_lmx_seq_len
        self.hidden_dim = hidden_dim

        with open(lmx_vocab_path, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        self.tokens_to_idxs = {token: i for i, token in enumerate(tokens)}
        self.idxs_to_tokens = {i: token for i, token in enumerate(tokens)}
        self.pad_idx = self.tokens_to_idxs[LMX_PAD_TOKEN]
        self.bos_idx = self.tokens_to_idxs[LMX_BOS_TOKEN]
        self.eos_idx = self.tokens_to_idxs[LMX_EOS_TOKEN]
        self.num_heads = num_heads
        self.head_dim = hidden_dim / num_heads
        self.vocab_size = len(tokens)

        self.vocab_embedding = nn.Embedding(
            self.vocab_size, self.hidden_dim, padding_idx=self.pad_idx
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(self.max_lmx_seq_len, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.1)

        self.decoder_blocks = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=transformer_dropout, activation="gelu", batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim, eps=1e-6)
        )

        self.unembed = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, input_seqs, img_latent, lmx_attention_mask, latent_attention_mask, token_idxs_input=True, checkpoint_grads=False):
        """
        input_seqs: 
            if token_idxs_input = True: (B, L_lmxmax), batch tensor of input sequences where each sequence is an 
                int tensor of a right-shifted original sequence
            if token_idxs_input = False: (B, L_lmxmax, E), batch tensor of input sequences of embeddings in vocab space. 
                Important for scheduled sampling where we need to mix embeddings and input those instead of hard indices
        img_latent: (B, L_imgmax, E) tensor of B image latents corresponding to each of the B input_seqs
        lmx_attention_mask: (B, L_lmxmax), records which lmx tokens are pad tokens
        latent_attention_mask: (B, L_imgmax), records which encoded patches in image batch are padding
        """
        batch_max_lmx_seq_len = input_seqs.shape[1] 
        if batch_max_lmx_seq_len > self.max_lmx_seq_len:
                raise ValueError(f"{batch_max_lmx_seq_len} long lmx sequence length is too long for max sequence length of {self.max_lmx_seq_len}")
        if token_idxs_input:
            lmx_embeddings = self.vocab_embedding(input_seqs) # (B, L_lmxmax, E)
        else:
            lmx_embeddings = input_seqs 

        # positionally embed lmx tokens
        pos_embed_slice = self.pos_embedding[:batch_max_lmx_seq_len, :]
        lmx_embeddings = lmx_embeddings + pos_embed_slice.unsqueeze(0)

        causal_mask = torch.triu(torch.ones(batch_max_lmx_seq_len, batch_max_lmx_seq_len), diagonal=1).bool() # enforce autoregressive predictions
        causal_mask = causal_mask.to(lmx_embeddings.device)
        if checkpoint_grads:
            # checkpointing only accepts positional args, so we pass kwargs to each layer's forward using a partial storing the kwargs. We use
            # layer-wise checkpointing to ensure we only ever are storing the intermediate activations for one decoder layer at a time (biggest savings)
            decoder_blocks = [
                partial(decoder_block, memory=img_latent, tgt_mask=causal_mask, tgt_key_padding_mask=lmx_attention_mask, memory_key_padding_mask=latent_attention_mask)
                for decoder_block in list(self.decoder_blocks.layers)
            ]
            hidden_state = checkpoint_sequential(decoder_blocks, segments=self.decoder_blocks.num_layers, input=lmx_embeddings, use_reentrant=False)
            hidden_state = self.decoder_blocks.norm(hidden_state)
        else:
            hidden_state = self.decoder_blocks(lmx_embeddings, memory=img_latent, tgt_mask=causal_mask, tgt_key_padding_mask=lmx_attention_mask, memory_key_padding_mask=latent_attention_mask)

        pred = self.unembed(hidden_state)
        return pred #(B, L_lmxmax, vocab_size)

    # minimal wrapper for batch inference: given a padded batch tensor of input_seqs and corresponding img_latent (optionally 
    # with a latent padding mask), generate next predictions for each sequence at each token position.
    def generate(self, input_seqs, img_latent, latent_attention_mask=None):
        seq_len = input_seqs.shape[1] 
        if seq_len > self.max_lmx_seq_len:
                raise ValueError(f"{seq_len} long lmx sequence length is too long for max sequence length of {self.max_lmx_seq_len}")

        lmx_embeddings = self.vocab_embedding(input_seqs)

        pos_embed_slice = self.pos_embedding[:seq_len, :]
        lmx_embeddings = lmx_embeddings + pos_embed_slice.unsqueeze(0)

        hidden_state = self.decoder_blocks(lmx_embeddings, memory=img_latent, memory_key_padding_mask=latent_attention_mask)
        pred = self.unembed(hidden_state)
        return pred

# lmx_seqs is a list of int tensors containing the unmodified lmx sequences associated with the input images, # in lmx token indices
def batchify_and_split_lmx_seqs(lmx_seqs: tuple[torch.Tensor], pad_idx, device):
    # split lmx sequences into right-shifted inputs and left-shifted targets
    lmx_seqs = torch.nested.as_nested_tensor(lmx_seqs)
    lmx_seqs = lmx_seqs.to_padded_tensor(padding=pad_idx)
    input_seqs = lmx_seqs[:, :-1]
    target_seqs = lmx_seqs[:, 1:]

    lmx_attention_mask = (input_seqs == pad_idx) # (B, L_lmxmax), True means that lmx token is a pad token
    lmx_attention_mask = lmx_attention_mask.to(device)
    return input_seqs, target_seqs, lmx_attention_mask

class TeacherForcedViTOMR(nn.Module):
    # create a model using parts from a pretrained MAE
    def __init__(self, omr_encoder, pretrained_mae_state_dict, omr_decoder, transition_head_dim=4096, transition_head_dropout=0.05):
        super().__init__()
        self.encoder = omr_encoder

        encoder_state_dict = self.create_omr_encoder_state_dict_from_mae(pretrained_mae_state_dict)
        self.encoder.load_state_dict(encoder_state_dict)

        # depending on the type of encoder being used, either freeze the whole thing or just the frozen blocks or none at all (for full fine-tune)
        if isinstance(self.encoder, FineTuneOMREncoder) and self.encoder.frozen_blocks:
            for param in self.encoder.frozen_blocks.parameters():
                param.requires_grad = False
            for param in self.encoder.projection.parameters():
                param.requires_grad = False
            self.encoder.pos_embedding.requires_grad = False
        elif isinstance(self.encoder, OMREncoder) and not isinstance(self.encoder, FineTuneOMREncoder):
            for param in self.encoder.parameters():
                param.requires_grad = False
        # for full fine-tune, we leave the state_dict as is (everything has requires_grad)

        self.decoder = omr_decoder

        # transition head to allow some task-specific adaptation of the latent representation
        self.transition_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, transition_head_dim),
            nn.GELU(),
            nn.Dropout(transition_head_dropout),
            nn.Linear(transition_head_dim, self.decoder.hidden_dim)
        )

    def create_omr_encoder_state_dict_from_mae(self, pretrained_mae_state_dict):
        # preprocess state dict: remove redundant prefixes added by MAE class since 
        pretrained_mae_state_dict = {
            param[len("encoder."):]: value for param, value in pretrained_mae_state_dict.items() if param.startswith("encoder.")
        }

        # the base OMREncoder still has an encoder_blocks instance variable, so the basic name adjustment is all 
        # that's needed to align the state dicts 
        if not isinstance(self.encoder, FineTuneOMREncoder): 
            return pretrained_mae_state_dict

        # separate encoder layers into frozen or fine-tune blocks depending on their depth
        omr_encoder_state_dict = {}
        freeze_threshold_depth = self.encoder.num_layers - self.encoder.fine_tune_depth
        last_norm_layer_parameters = ["encoder_blocks.norm.weight", "encoder_blocks.norm.bias"]
        layer_num_pattern = re.compile(r"(?:\w|\.)+?(\d+)(?:\w|\.)+")
        for param in pretrained_mae_state_dict.keys():
            if match := layer_num_pattern.match(param): # this is a layer in MAEEncoder's encoder_blocks variable
                layer_num = int(match.group(1))
                if layer_num < freeze_threshold_depth:
                    new_param_name = param.replace("encoder_blocks", "frozen_blocks")
                else:
                    new_param_name = param.replace("encoder_blocks", "fine_tune_blocks")
                    # shift layer numbers to start at index 0 in this new block
                    new_param_name = new_param_name.replace(f"layers.{layer_num}", f"layers.{layer_num - freeze_threshold_depth}")
                omr_encoder_state_dict[new_param_name] = pretrained_mae_state_dict[param]

            elif param in last_norm_layer_parameters: # the last norm layer should be fine tunable
                new_param_name = param.replace("encoder_blocks", "fine_tune_blocks")
                omr_encoder_state_dict[new_param_name] = pretrained_mae_state_dict[param]

            else: # parameters that are shared between all these Encoder types (eg self.projection)
                omr_encoder_state_dict[param] = pretrained_mae_state_dict[param]
        
        return omr_encoder_state_dict
   

    """
    Input
        x: a list of (image, lmx_sequence) tuples where each lmx_sequence is an int tensor of input token indices, with <bos> and <eos> tokens added to each already
    Output
        pred: (B, L_lmxmax, vocab_size)
        target_seqs: (B, L_lmxmax)
    """
    def forward(self, x: list[tuple[torch.Tensor, torch.Tensor]]):
        imgs, lmx_seqs = zip(*x)

        # encode images
        img_latent, latent_attention_mask = self.encoder(imgs)        

        # feed through head
        img_latent = self.transition_head(img_latent)

        # prepare lmx sequences and mask
        input_seqs, target_seqs, lmx_attention_mask = batchify_and_split_lmx_seqs(lmx_seqs, self.decoder.pad_idx, img_latent.device)

        # decode lmx with cross-attention to image latent
        pred = self.decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask)
        return pred, target_seqs # return target_seqs for later use in loss

    """
    Input
        img_latent: a (1, T, E_dec) latent representation of the image to run inference on (must have already been fed through the transition head)
        seqs: a (beam_width, seq_len) tensor of the sequences to append to. Unlike the regular forward method used for training,
        for this method at the start each sequence should just be one <bos> token
    Output
        next_token_distr: a (beam_width, vocab_size) tensor of the probability distribution (in log probs) for the next token for each
        sequence
    A general note: all these generate() methods are optimized for inference (mainly by cutting out all the extra work needed to
    deal with ragged image/sequence batches)
    """
    def generate(self, img_latent: torch.Tensor, seqs: torch.Tensor):
        num_seqs = seqs.shape[0]
        # expand() more memory efficient than repeat()
        img_latent = img_latent.expand(num_seqs, -1, -1)
        
        logits = self.decoder.generate(seqs, img_latent)
        next_token_distr = F.log_softmax(logits[:, -1, :], dim=-1)

        return next_token_distr

    # split each encoder attention block into a different parameter group and apply llrd. Encoder PE and projection to embedding space
    # will be given the same lr as the earliest attention block (since they're the earliest encoding operations)
    def create_fine_tune_param_groups(self, base_lr: float, fine_tune_base_lr: float, fine_tune_decay_factor: float):
        param_groups = [
            {"params": self.decoder.parameters(), "lr": base_lr}, 
            {"params": self.transition_head.parameters(), "lr": base_lr}, 
        ]
        layer_lrs = []
        for i, layer in enumerate(reversed(self.encoder.fine_tune_blocks.layers)):
            layer_lr = fine_tune_base_lr * (fine_tune_decay_factor ** i)
            param_groups.append({"params": layer.parameters(), "lr": layer_lr})
            layer_lrs.append(layer_lr)
        min_layer_lr = layer_lrs[-1]

        # add encoder params that aren't part of individual nn.TransformerEncoderLayers: pe grid, projection into embedding space, nn.TransformerEncoder final norm
        param_groups.append({"params": self.encoder.fine_tune_blocks.norm.parameters(), "lr": fine_tune_base_lr})
        # optimizer expects parameters to be passed in some kind of iterable (and not as nn.Parameter objects). Use a generator here for consistency with the others
        param_groups.append({"params": (param for param in [self.encoder.pos_embedding]), "lr": min_layer_lr}) 
        param_groups.append({"params": self.encoder.projection.parameters(), "lr": min_layer_lr})

        # IMPORTANT NOTE: these parameters are generators so they can only be consumed once. Eg if in testing code they're consumed
        # to be checked later, the optimizer will have no parameter generators to use to update the parameters
        return param_groups, layer_lrs

# wrapper to handle the logic with cross entropy loss
class OMRCELoss(nn.Module):
    def __init__(self, pad_idx, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=self.label_smoothing)

    def forward(self, pred, target_seqs):
        """
        pred: (B, L_lmxmax, vocab_size), decoder output logits
        target_seqs: (B, L_lmxmax), tokens to predict at each position
        """
        # flatten tensors since target tensor needs to be shape (N, )
        return self.loss_fn(pred.reshape(-1, pred.shape[-1]), target_seqs.reshape(-1))

class ScheduledSamplingViTOMR(TeacherForcedViTOMR):
    # mix teacher forced sequence with first pass predictions: at each time step i, if sample = True we use the first 
    # pass' predicted distribution for the ith token to create an expected embedding and use that as the second pass' ith input embedding
    def sample_and_mix_seqs(self, teacher_forcing_prob, tf_input_seqs, tf_pred_logits, sample_tau, use_hard_sampling, device):
        sampling_ratio = 1 - teacher_forcing_prob
        sample_mask = torch.rand(tf_input_seqs.shape, device=device) < sampling_ratio # (B, T)

        gold_token_embeddings = self.decoder.vocab_embedding(tf_input_seqs)

        # create expected embeddings at each token position using teacher-forced pass predictions
        tf_pred_distrs = F.gumbel_softmax(tf_pred_logits, tau=sample_tau, hard=use_hard_sampling) # (B, T, V)
        tf_expected_embeddings = tf_pred_distrs @ self.decoder.vocab_embedding.weight # (B x T x V) x (V x E) = (B x V x E)

        # rightshift predicted sequence to align it with rightshifted teacher forced inputs for sampling
        bos_stem = gold_token_embeddings[:, 0:1, :]
        tf_expected_embeddings = torch.cat([bos_stem, tf_expected_embeddings], dim=1)
        tf_expected_embeddings = tf_expected_embeddings[:, :-1]

        mixed_input_seqs = torch.where(sample_mask.unsqueeze(-1), tf_expected_embeddings, gold_token_embeddings)
        return mixed_input_seqs

    def forward_train(self, x: list[tuple[torch.Tensor, torch.Tensor]], teacher_forcing_prob: float, sample_tau: float, use_hard_sampling: bool):
        imgs, lmx_seqs = zip(*x)

        img_latent, latent_attention_mask = self.encoder(imgs)
        img_latent = self.transition_head(img_latent)
        device = img_latent.device

        tf_input_seqs, target_seqs, lmx_attention_mask = batchify_and_split_lmx_seqs(lmx_seqs, self.decoder.pad_idx, device)

        # first decoder pass: generate logits from completely gold inputs
        tf_pred_logits = self.decoder(tf_input_seqs, img_latent, lmx_attention_mask, latent_attention_mask) # (B, T, V)

        mixed_input_seqs = self.sample_and_mix_seqs(teacher_forcing_prob, tf_input_seqs, tf_pred_logits, sample_tau, use_hard_sampling, device)

        pred = self.decoder(mixed_input_seqs, img_latent, lmx_attention_mask, latent_attention_mask, token_idxs_input=False)
        return pred, target_seqs # target sequences for loss calculation remain the same

    # for validation/evaluation, we stick with regular teacher forcing
    def forward_eval(self, x: list[tuple[torch.Tensor, torch.Tensor]]):
        return super().forward(x)

class GRPOViTOMR(nn.Module):
    # encoder, transition_head, and decoder should be module instances matching those of a TeacherForcedViTOMR 
    # instance. init essentially will convert that instance into a version prepared for GRPO training
    def __init__(self, encoder, transition_head, decoder, teacher_forced_state_dict):
        super().__init__()
        if isinstance(encoder, FineTuneOMREncoder):
            teacher_forced_state_dict = self.convert_teacher_forced_state_dict(teacher_forced_state_dict, encoder.num_frozen_layers)
            encoder = OMREncoder(encoder.patch_size, encoder.pe_max_height, encoder.pe_max_width, encoder.num_layers, encoder.hidden_dim, **encoder.superclass_kwargs)
            
        self.encoder = encoder
        self.transition_head = transition_head
        self.decoder = decoder
        self.load_state_dict(teacher_forced_state_dict)
        # freeze encoder/transition head which by now should be well-trained for the seq2seq task. Also disable their dropout
        self.freeze_component(self.encoder)
        self.freeze_component(self.transition_head)

    def freeze_component(self, component):
        for param in component.parameters():
            param.requires_grad = False
        for child in component.modules():
            if isinstance(child, nn.Dropout):
                child.p = 0.0

    def convert_teacher_forced_state_dict(self, teacher_forced_state_dict, num_frozen_layers):
        converted_state_dict = {} # new dict since can't mutate old dict while iterating over it

        layer_num_pattern = re.compile(r"(?:\w|\.)+?(\d+)(?:\w|\.)+")
        for param in teacher_forced_state_dict.keys():
            if "frozen_blocks" in param:
                new_param_name = param.replace("frozen_blocks", "encoder_blocks")
            # fine-tune blocks have shifted layer numbers for partial fine-tune
            elif "fine_tune_blocks" in param:
                new_param_name = param.replace("fine_tune_blocks", "encoder_blocks")
                # no match means this is a parameter for the last norm of the whole encoder, in which case
                # the name replacement was enough
                if match := layer_num_pattern.match(param):
                    layer_num = int(match.group(1))
                    new_param_name = new_param_name.replace(f"layers.{layer_num}", f"layers.{layer_num + num_frozen_layers}")
            else:
                new_param_name = param

            converted_state_dict[new_param_name] = teacher_forced_state_dict[param]
        return converted_state_dict

    # expand latent/mask to create multiple rollouts for each example
    def expand_img_latent_for_rollout(self, img_latent, latent_attention_mask, group_size):
        img_latent = img_latent.unsqueeze(1)
        img_latent = img_latent.expand(-1, group_size, -1, -1) # (B, group_size, T, E_enc)
        img_latent = img_latent.flatten(start_dim=0, end_dim=1) # (B x group_size, T, E_enc)

        latent_attention_mask = latent_attention_mask.unsqueeze(1)
        latent_attention_mask = latent_attention_mask.expand(-1, group_size, -1)
        latent_attention_mask = latent_attention_mask.flatten(start_dim=0, end_dim=1)
        return img_latent, latent_attention_mask

    # returns mask where True = token is part of a rollout, False = token is junk after a rollout ended
    def create_rollout_mask(self, rollouts):
        # marks where <eos> tokens are
        eos_mask = (rollouts == self.decoder.eos_idx)
        # 0 when there's been no <eos> up to/including this point, 1 when there's been 1 up to/including this point, etc.
        seen_eos_mask = eos_mask.int().cumsum(dim=-1)
        # first <eos> is when we have an <eos> token and have only seen 1 so far (ie at that position)
        first_eos_mask = eos_mask & (seen_eos_mask == 1)
        # only count each rollout as tokens up to and including the first <eos> (if it exists and if include_eos = True)
        rollout_mask = (seen_eos_mask == 0) | first_eos_mask
        return rollout_mask

    """
    Inputs
        img_latent: (R, T, E_enc) batched, padded tensor of B image latents each duplicated across group_size rollouts into 
        B x group_size = R total rows
        latent_attention_mask: (R, T) mask showing what embeddings in img_latent are padding
        Note that this can also be used for batched evaluation/inference: simply pass in an unexpanded img_latent and mask to get one
        rollout per example
    Outputs
        rollouts: (R, T) padded tensor of rollouts. May contain junk if some sequences terminated early
        rollout_log_probs: (R, T) tensor of the log prob for choosing the chosen token at each step of rollouts
        rollout_mask: (R, T) tensor where True = token is part of a rollout, False = token isn't
    For each (T, E_enc) image latent in img_latent, create group_size autoregressive rollouts according to the
    policy defined by the model's outputted distributions, up to max_actions steps in total (where <bos> stems count as 1 action) for each. 
    At each autoregressive step, set logits that aren't in the top_k top logits to -inf, then apply softmax with temperature, 
    then extend the sequence according to the resulting distribution 
    Note: predictions after first train seem pretty peaky so default is to use temperature > 1 to slightly smooth things and 
    encourage more exploration.
    Also note that this should be called with torch_no_grad by the old policy and img_latent and latent_attention_mask should
    be the result of self.expand_img_tensors_for_rollout
    TODO: change this so keeps track of an active mask so only keep doing inference on active seqs, fill rest with <pad> and 0 log prob for next token
    """
    def uncached_forward_rollout_policy(self, img_latent, latent_attention_mask, max_actions=768, top_k=50, temperature=1.2):
        device = img_latent.device

        total_rollouts = img_latent.shape[0]
        # for efficiency, preallocate tensors to later fill in by indexing in place
        rollouts = torch.full([total_rollouts, max_actions + 1], fill_value=self.decoder.pad_idx, dtype=torch.long, device=device)
        rollouts[:, 0] = torch.full([total_rollouts], fill_value=self.decoder.bos_idx)

        # per-token log probs help us later mask out junk parts of sequences/calculating GRPO objective per step. Use log-probs
        # for numerical stability, eg with super small regular probs
        rollout_log_probs = torch.zeros_like(rollouts, dtype=torch.float, device=device)
        for t in range(1, max_actions):
            prefix = rollouts[:, :t] # preallocated tensors to max_actions, so only pass in the prefix so far 
            logits = self.decoder.generate(prefix, img_latent, latent_attention_mask=latent_attention_mask)
            logits = logits[:, -1, :] # (R x E_dec), next token distributions for each rollout

            # top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            top_k_mask = torch.full_like(logits, float("-inf"))
            logits = top_k_mask.scatter(-1, top_k_indices, top_k_logits)

            # extend sequences
            softmax_logits = logits / temperature # apply temperature for softmax
            next_token_distr = F.softmax(softmax_logits, dim=-1)
            next_token_idxs = torch.multinomial(next_token_distr, num_samples=1) # (R, 1)
            rollouts[:, t] = next_token_idxs.squeeze(1)

            token_log_probs = F.log_softmax(softmax_logits, dim=-1)
            next_token_log_probs = token_log_probs.gather(-1, index=next_token_idxs)
            rollout_log_probs[:, t] = next_token_log_probs.squeeze(1)

            # end early if all seqs are done early
            finished_seqs = torch.any((rollouts == self.decoder.eos_idx), dim=-1)
            if torch.all(finished_seqs):
                break
        
        rollout_mask = self.create_rollout_mask(rollouts)
        # explicitly set tokens that aren't part of rollouts to <pad> and their log probs to 0. Later functions assume this is the case
        rollouts = rollouts.masked_fill(~rollout_mask, self.decoder.pad_idx)
        rollout_log_probs = rollout_log_probs.masked_fill(~rollout_mask, 0.0)

        return rollouts, rollout_log_probs, rollout_mask

    # right shift rollouts and prepare an attention mask for it (since it's likely a padded tensor of ragged rollouts)
    def prepare_rollouts_for_policy_theta(self, rollouts, rollout_mask):
        # can't directly use rollout_mask as policy theta's rollout_attention_mask because it may include raggedly-placed
        # <eos>'s as part of rollouts which we don't want included in our right shifted inputs
        rollout_lens = rollout_mask.sum(dim=-1, keepdim=True)
        right_shifted_rollout_lens = rollout_lens - 1 
        rollout_attention_mask = torch.arange(torch.max(right_shifted_rollout_lens), device=rollouts.device).repeat([rollouts.shape[0], 1])
        rollout_attention_mask = rollout_attention_mask >= right_shifted_rollout_lens
        right_shifted_rollouts = rollouts[:, :-1]
        return right_shifted_rollouts, rollout_attention_mask

    # img_latent and latent_attention_mask should be the unexpanded results of an encoder pass so we can
    # avoid unnecessarily recalculating them here. lmx_seqs should be the targets we're running the teacher forced
    # pass on (again, this hasn't been expanded so we can save on computation)
    def forward_teacher_forced(self, img_latent, latent_attention_mask, lmx_seqs: list[torch.Tensor], checkpoint_grads):
        input_seqs, target_seqs, lmx_attention_mask = batchify_and_split_lmx_seqs(lmx_seqs, self.decoder.pad_idx, img_latent.device)

        pred = self.decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask, checkpoint_grads=checkpoint_grads)
        return pred, target_seqs 

    # wrapper that takes a batch of image tensors, encodes them, and runs 1 rollout on each example. Should be used for 
    # evaluation/inference, ie wrapped in no_grad()
    def batch_policy_inference(self, imgs: list[torch.Tensor], max_actions, top_k, temperature):
        img_latent, latent_attention_mask = self.encoder(imgs)
        rollouts, rollout_log_probs, rollout_mask = self.forward_rollout_policy(img_latent, latent_attention_mask, max_actions, top_k, temperature)
        return rollouts, rollout_log_probs, rollout_mask

    # version with KV caching
    def cached_forward_rollout_policy(self, img_latent, latent_attention_mask, max_actions=768, top_k=50, temperature=1.2):
        kv_cache = KVCache(batch_size=img_latent.shape[0], max_seq_len=max_actions, num_kv_heads=self.decoder.num_heads, head_dim=self.decoder.head_dim, dtype=torch.float)
        pass
