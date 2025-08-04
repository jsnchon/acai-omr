import torch
from torch import nn
from config import LMX_PAD_TOKEN

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
        src_key_padding_mask = self.create_attention_mask(seq_lens, embeddings.shape[1]).to(embeddings.device)
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
        return x, src_key_padding_mask # return mask for later use in loss calculation and cross-attention masking

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

    def batchify(self, x: list[torch.Tensor]):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        seq_lens = []
        pos_embed_slices = []
        patchified_tensors = []
        for t in x:
            h_p = t.shape[-2] // self.patch_size
            w_p = t.shape[-1] // self.patch_size

            t = unfold(t.unsqueeze(0)) 
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

class OMRDecoder(nn.Module):
    def __init__(self, max_lmx_seq_len, lmx_vocab_path, num_layers=6, hidden_dim=1024, num_heads=16, mlp_dim=4096, transformer_dropout=0.1):
        super().__init__()
        self.max_lmx_seq_len = max_lmx_seq_len
        self.hidden_dim = hidden_dim

        with open(lmx_vocab_path, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        self.tokens_to_idxs = {token: i for i, token in enumerate(tokens)}
        self.idxs_to_tokens = {i: token for i, token in enumerate(tokens)}
        self.padding_idx = self.tokens_to_idxs[LMX_PAD_TOKEN]
        self.vocab_size = len(tokens)

        self.vocab_embedding = nn.Embedding(
            self.vocab_size, self.hidden_dim, padding_idx=self.padding_idx
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

    def forward(self, input_seqs, img_latent, lmx_attention_mask, latent_attention_mask):
        """
        input_seqs: (B, L_lmxmax), batch tensor of input sequences where each sequence is an int tensor of a right-shifted original sequence
        img_latent: (B, L_imgmax, E)
        lmx_attention_mask: (B, L_lmxmax), records which lmx tokens are pad tokens
        latent_attention_mask: (B, L_imgmax), records which encoded patches in image batch are padding
        """
        batch_max_lmx_seq_len = input_seqs.shape[1] 
        if batch_max_lmx_seq_len > self.max_lmx_seq_len:
                raise ValueError(f"{batch_max_lmx_seq_len} long lmx sequence length is too long for max sequence length of {self.max_lmx_seq_len}")
        lmx_embeddings = self.vocab_embedding(input_seqs) # (B, L_lmxmax, E)

        # positionally embed lmx tokens
        pos_embed_slice = self.pos_embedding[:batch_max_lmx_seq_len, :]
        lmx_embeddings = lmx_embeddings + pos_embed_slice.unsqueeze(0)

        causal_mask = torch.triu(torch.ones(batch_max_lmx_seq_len, batch_max_lmx_seq_len), diagonal=1).bool() # enforce autoregressive predictions
        causal_mask = causal_mask.to(lmx_embeddings.device)
        hidden_state = self.decoder_blocks(lmx_embeddings, memory=img_latent, tgt_mask=causal_mask, tgt_key_padding_mask=lmx_attention_mask, memory_key_padding_mask=latent_attention_mask)

        pred = self.unembed(hidden_state)
        return pred

class ViTOMR(nn.Module):
    # masking logic for image encodings and padded LMX token sequences. Add prepare_for_decoder method to do this?
    
    # create embedding param for token vocab
    def __init__(self, omr_encoder, pretrained_mae_state_dict, omr_decoder, transition_head_dim=4096, dropout_p=0.1):
        super().__init__()
        self.encoder = omr_encoder

        # extract encoder parameters and remove redundant prefixes added by MAE class to align names
        encoder_state_dict = {
            param[len("encoder."):]: value for param, value in pretrained_mae_state_dict.items() if param.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state_dict)

        # freeze the entire encoder 
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = omr_decoder

        # transition head to allow some task-specific adaptation of the latent representation
        self.transition_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, transition_head_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(transition_head_dim, self.decoder.hidden_dim)
        )
    
    # lmx_seqs is a list of int tensors containing the unmodified lmx sequences associated with the input images,
    # in lmx token indices
    def batchify_and_split_lmx_seqs(self, lmx_seqs: list[torch.Tensor], device):
        # split lmx sequences into right-shifted inputs and left-shifted targets
        padding_idx = self.decoder.padding_idx
        lmx_seqs = torch.nested.as_nested_tensor(lmx_seqs)
        lmx_seqs = lmx_seqs.to_padded_tensor(padding=padding_idx)
        input_seqs = lmx_seqs[:, :-1]
        target_seqs = lmx_seqs[:, 1:]

        lmx_attention_mask = input_seqs == padding_idx # (B, L_lmxmax), True means that lmx token is a pad token
        lmx_attention_mask = lmx_attention_mask.to(device)
        return input_seqs, target_seqs, lmx_attention_mask

    # x is a list of (image, lmx_sequence) tuples where each lmx_sequence is an int tensor of input token indices
    def forward(self, x: list[tuple[torch.Tensor, torch.Tensor]]):
        imgs, lmx_seqs = zip(*x)
        imgs = list(imgs)
        lmx_seqs = list(lmx_seqs)

        # encode images
        img_latent, latent_attention_mask = self.encoder(imgs)        

        # feed through head
        img_latent = self.transition_head(img_latent)

        # prepare lmx sequences and mask
        input_seqs, target_seqs, lmx_attention_mask = self.batchify_and_split_lmx_seqs(lmx_seqs, img_latent.device)

        # decode lmx with cross-attention to image latent
        pred = self.decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask)
        return pred, target_seqs # return target_seqs for later use in loss

# wrapper to handle the logic with cross entropy loss
class OMRLoss(nn.Module):
    def __init__(self, padding_idx, label_smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=label_smoothing)

    def forward(self, pred, target_seqs):
        """
        pred: (B, L_lmxmax, vocab_size), decoder output
        target_seqs: (B, L_lmxmax), tokens to predict at each position
        """
        # flatten tensors since target tensor needs to be shape (N, )
        return self.loss_fn(pred.reshape(-1, pred.shape[-1]), target_seqs.reshape(-1))
