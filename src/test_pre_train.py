import torch
from torch import nn
from models import Encoder, MAEEncoder, MAE, MAELoss

def test_encoder_batchify():
    encoder = Encoder(patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    batch, mask = encoder.batchify(x)
    print(batch)
    print(mask)
    first_ex_mask = (torch.arange(8) >= 4).unsqueeze(0)
    second_ex_mask = (torch.arange(8) >= 8).unsqueeze(0)
    assert torch.equal(mask, torch.cat((first_ex_mask, second_ex_mask)))

def test_encoder_forward():
    hidden_dim = 200
    encoder = Encoder(num_layers=2, num_heads=2, hidden_dim=hidden_dim, mlp_dim=500, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    x, mask = encoder(x)
    print(x)
    print(mask)
    assert x.shape == torch.Size([2, 8, hidden_dim])

def test_mask_sequence():
    encoder = MAEEncoder(0.50, num_heads=1, hidden_dim=1)
    SEQ_LEN = 4
    x = torch.arange(SEQ_LEN).unsqueeze(0).repeat(12, 1).unsqueeze(0) # simulate patch size 2 (so CP^2 = 12) w/ sequence of length 4, label patches to check shuffle/unshuffle
    print(f"x before shuffle/mask: {x}, shape: {x.shape}")
    pe_num_grid = torch.arange(4, dtype=torch.float).reshape(2, 2) # pes labeled in order to check alignment with shuffle
    pe_filler = torch.zeros(2, 4) - 1 # should not appear in final slice to be added to embeddings
    encoder.pos_embedding = nn.Parameter(
        torch.cat((pe_num_grid, pe_filler), dim=1).unsqueeze(-1)
    )
    print(f"Unsliced positional embedding grid: {encoder.pos_embedding}")
    t_masked, pos_embed_slice, unmasked_seq_len, len_keep, seq_mask, ids_restore = encoder.mask_sequence(x, 2, 2)
    print(f"Output:\nt_masked: {t_masked}, shape: {t_masked.shape}\npos_embed_slice: {pos_embed_slice}, shape: {pos_embed_slice.shape}\nunmasked_seq_len: {unmasked_seq_len}\nlen_keep: {len_keep}\nseq_mask: {seq_mask}\nids_restore: {ids_restore}")
    assert t_masked.shape == torch.Size([1, 12, 2])
    assert unmasked_seq_len == SEQ_LEN
    assert len_keep == 2
    t_masked = torch.concat((t_masked, (torch.zeros(1, 12, 3) - 1)), dim=-1) # append mask tokens of -1 tensors
    undo = t_masked.index_select(dim=-1, index=ids_restore.squeeze(0)) 
    print(f"After appending mask tokens and unshuffling: {undo}")
    assert undo.shape == x.shape

def test_masked_encoder_batchify():
    encoder = MAEEncoder(0.50, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 6)]
    batch, encoder_attn_mask, decoder_attn_mask, seq_masks, ids_restores = encoder.batchify(x)
    print(f"Output:\nEmbeddings {batch}\nEncoder attention mask: {encoder_attn_mask}\nDecoder attention mask: {decoder_attn_mask}\\nSequence masks: {seq_masks}\nRestore tensor: {ids_restores}")
    # verify encoder attention mask
    first_ex_mask = (torch.arange(3) >= 2).unsqueeze(0)
    second_ex_mask = (torch.arange(3) >= 3).unsqueeze(0)
    assert torch.equal(encoder_attn_mask, torch.cat((first_ex_mask, second_ex_mask)))
    # verify decoder attention mask
    first_ex_mask = (torch.arange(6) >= 4).unsqueeze(0)
    second_ex_mask = (torch.arange(6) >= 6).unsqueeze(0)
    assert torch.equal(decoder_attn_mask, torch.cat((first_ex_mask, second_ex_mask)))

def test_masked_encoder_forward():
    hidden_dim = 200
    encoder = MAEEncoder(0.50, num_layers=2, num_heads=2, hidden_dim=hidden_dim, mlp_dim=500, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    x, attn_mask, _, seq_masks, ids_restores = encoder(x)
    print(f"Output:\nEmbeddings {x}\nAttention mask: {attn_mask}\nSequence masks: {seq_masks}\nRestore tensor: {ids_restores}")
    assert x.shape == torch.Size([2, 4, hidden_dim])

def test_prepare_for_decoder():
    encoder_kwargs = {"num_heads": 1}
    decoder_kwargs = {"num_heads": 1}
    encoder_hidden_dim = 2
    mae = MAE(0.5, 1, encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=1, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)
    # simulate latent of two sequences, one with length 2 + padding and other with length 3. First needs 2 mask tokens appended, second
    # needs 3 mask tokens appended. When unshuffled, mask tokens should be in front, labeled patches should be in ascending order
    mae.mask_token = nn.Parameter(torch.zeros(1, 1, 1) + 100)
    first_latent_seq = torch.cat([(torch.arange(2) + 1).unsqueeze(-1).unsqueeze(0), torch.zeros(1, 1, 1) - 1], dim=1) 
    second_latent_seq = (torch.arange(3) + 1).unsqueeze(-1).unsqueeze(0)
    # shuffle
    first_latent_seq = first_latent_seq.index_select(dim=1, index=torch.tensor([1, 0, 2]))
    second_latent_seq = second_latent_seq.index_select(dim=1, index=torch.tensor([2, 0, 1]))
    kept_seq_lens = [2, 3]
    unmasked_seq_lens = [4, 6]
    patchified_dims = [(2, 2), (2, 3)] # simulate original images being 2 x 2, 2 x 3 patches
    batch_ids_restore = torch.nested.nested_tensor([torch.tensor([2, 3, 1, 0]), torch.tensor([3, 4, 5, 1, 2, 0])])
    latent = torch.cat([first_latent_seq, second_latent_seq])
    print(f"Latent before reconstruction (shuffled and padded from encoder): {latent}, {latent.shape}")

    pe_num_grid = torch.zeros(2, 3) + 500
    pe_filler = torch.zeros(2, 4) - 1 # should not appear in final slice to be added to embeddings
    mae.decoder_pos_embedding = nn.Parameter(
        torch.cat((pe_num_grid, pe_filler), dim=1).unsqueeze(-1)
    )
    print(f"Unsliced decoder positional embedding grid: {mae.decoder_pos_embedding}")

    reconstructed_seq = mae.prepare_for_decoder(latent, kept_seq_lens, unmasked_seq_lens, batch_ids_restore, patchified_dims)
    print(f"Latent after reconstruction: {reconstructed_seq}, {reconstructed_seq.shape}")
    first_target_seq = torch.cat([
        torch.tensor([100, 100, 1, 2]).unsqueeze(-1).unsqueeze(0) + 500, # unshuffled/positionally embedded part
        torch.tensor([0, 0]).unsqueeze(-1).unsqueeze(0)], dim=1) # padding part
    second_target_seq = torch.tensor([100, 100, 100, 1, 2, 3]).unsqueeze(-1).unsqueeze(0) + 500
    assert torch.equal(reconstructed_seq, torch.cat([first_target_seq, second_target_seq]))

def test_MAE():
    encoder_kwargs = {"num_heads": 1, "num_layers": 4}
    decoder_kwargs = {"num_heads": 1, "num_layers": 2}
    encoder_hidden_dim = 6
    decoder_hidden_dim = 4
    mae = MAE(0.5, 1, encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)
    SEQ_LEN = 4
    x = [torch.arange(SEQ_LEN, dtype=torch.float).reshape(2, 2).unsqueeze(0).repeat(3, 1, 1)] 
    print(x[0].shape)
    print(f"x before mae forward: {x}")
    pe_num_grid = torch.arange(4, dtype=torch.float).reshape(2, 2) # pes labeled in order to check alignment with shuffle
    pe_filler = torch.zeros(2, 4) - 1 # should not appear in final slice to be added to embeddings
    mae.encoder.pos_embedding = nn.Parameter(
        torch.cat((pe_num_grid, pe_filler), dim=1).unsqueeze(-1).repeat(1, 1, encoder_hidden_dim)
    )
    mae.decoder_pos_embedding = nn.Parameter(
        torch.cat((pe_num_grid, pe_filler), dim=1).unsqueeze(-1).repeat(1, 1, decoder_hidden_dim)
    )
    pred, loss_mask, target = mae(x)
    print(f"Prediction: {pred}\nLoss mask: {loss_mask}")

    x= [torch.arange(SEQ_LEN, dtype=torch.float).reshape(2, 2).unsqueeze(0).repeat(3, 1, 1),
        torch.arange(SEQ_LEN * 2, dtype=torch.float).reshape(2, -1).unsqueeze(0).repeat(3, 1, 1)]
    print(f"x before mae forward: {x}")
    pred, loss_mask, _ = mae(x)
    print(f"Prediction: {pred}\nLoss mask: {loss_mask}")
    first_seq_loss_mask = loss_mask[0, :]
    second_seq_loss_mask = loss_mask[1, :]

    assert torch.sum(first_seq_loss_mask[SEQ_LEN:]) == 0 # last half should definitely be False since it was padding for attention
    # half of original sequence length should be True
    assert torch.sum(first_seq_loss_mask) == SEQ_LEN / 2
    assert torch.sum(second_seq_loss_mask) == SEQ_LEN 

def test_MAE_loss():
    loss = MAELoss()
    target = torch.cat([
        torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float).unsqueeze(-1).repeat(1, 1, 6),
        torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.float).unsqueeze(-1).repeat(1, 1, 6),
        ], dim=-1)
    pred = torch.tensor([[2, 2, 2], [3, 3, 4]], dtype=torch.float).unsqueeze(-1).repeat(1, 1, 12)
    loss_mask = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=torch.float)
    print(f"Target:\n{target}\nPrediction:\n{pred}\nLoss mask\n{loss_mask}")
    loss = loss(pred, loss_mask, target)
    print(f"Loss:\n{loss}")
    assert loss == 10.583329200744629

if __name__ == "__main__":
    test_MAE_loss()