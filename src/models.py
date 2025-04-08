import torch
from torch import nn

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