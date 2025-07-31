import torch
from torch import nn
from models import OMREncoder, OMRDecoder, ViTOMR, NUM_CHANNELS
from pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH
from omr_train import MAX_LMX_SEQ_LEN
from utils import DEBUG_PRETRAINED_MAE_PATH

debug_patch_size = 2
debug_encoder_kwargs = {"num_layers": 1, "num_heads": 1, "hidden_dim": 1, "mlp_dim": 1}
debug_encoder = OMREncoder(debug_patch_size, PE_MAX_HEIGHT, PE_MAX_WIDTH, **debug_encoder_kwargs)
# the encoder structure used in the pre_train loop test
pretrained_debug_encoder = OMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, **debug_encoder_kwargs)
 
def test_encoder_batchify():
    patch_size = 2
    hidden_dim = NUM_CHANNELS * patch_size ** 2
    encoder = OMREncoder(patch_size, PE_MAX_HEIGHT, PE_MAX_WIDTH, hidden_dim=hidden_dim, num_heads=1)
    encoder.projection = nn.Identity()
    encoder.pos_embedding = nn.Parameter(torch.ones(50, 50, hidden_dim))

    x = [torch.ones(NUM_CHANNELS, 4, 4), torch.ones(NUM_CHANNELS, 4, 8)]
    embeddings, mask = encoder.batchify(x)
    print(f"Output:\nEmbeddings:\n{embeddings}\nMask:\n{mask}")
    first_embedding_seq = torch.cat([torch.ones(NUM_CHANNELS, 4, hidden_dim) + 1, torch.zeros(NUM_CHANNELS, 4, hidden_dim)], dim=1)
    second_embedding_seq = torch.ones(NUM_CHANNELS, 8, hidden_dim) + 1
    embeddings_target = torch.cat([first_embedding_seq, second_embedding_seq])
    assert torch.equal(embeddings_target, embeddings)
    first_ex_mask = (torch.arange(8) >= 4).unsqueeze(0)
    second_ex_mask = (torch.arange(8) >= 8).unsqueeze(0)
    assert torch.equal(mask, torch.cat((first_ex_mask, second_ex_mask)))

    # test larger inputs where pe needs to be interpolated
    x = [torch.ones(NUM_CHANNELS, 10, 600)]
    embeddings, mask = encoder.batchify(x) 
    # since PE grid is all ones, should be interpolated to a grid of all ones matching input image dims
    assert torch.equal(embeddings, torch.ones(NUM_CHANNELS, 1500, hidden_dim) + 1)

def test_encoder_forward():
    encoder = debug_encoder
    x = [torch.rand(NUM_CHANNELS, 4, 4), torch.rand(NUM_CHANNELS, 4, 8)]
    x, mask = encoder(x)
    print(x)
    print(mask)
    assert x.shape == torch.Size([2, 8, debug_encoder_kwargs["hidden_dim"]])

    # test larger inputs where pe needs to be interpolated
    x = [torch.rand(NUM_CHANNELS, 10, 600)]
    x, mask = encoder(x)
    print(x)
    print(mask)
    assert x.shape == torch.Size([NUM_CHANNELS, 1500, debug_encoder_kwargs["hidden_dim"]])

def test_decoder_batchify():
    hidden_dim = 1
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", hidden_dim=hidden_dim, num_heads=1)

    lmx_seqs = [torch.tensor([0, 2, 3, 226]), torch.tensor([0, 2, 2, 3, 4, 226])]
    padded_lmx_seqs, lmx_attention_mask, batch_max_seq_len = decoder.batchify_lmx_seqs(lmx_seqs)
    print(f"Padded sequences\n{padded_lmx_seqs}\nAttention mask\n{lmx_attention_mask}")
    assert padded_lmx_seqs.shape == torch.Size([2, 6])
    assert lmx_attention_mask.shape == torch.Size([2, 6])
    assert batch_max_seq_len == 6
    assert torch.equal(lmx_attention_mask[0, :], torch.tensor([False, False, False, False, True, True]))
    assert torch.equal(lmx_attention_mask[1, :], torch.tensor([False, False, False, False, False, False]))

def test_decoder():
    VOCAB_LEN = 227
    hidden_dim = 10
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", hidden_dim=hidden_dim, num_heads=1, num_layers=1, mlp_dim=1)

    # test vocabulary creation
    assert decoder.vocab_embedding.weight.shape == torch.Size([VOCAB_LEN, hidden_dim])

    # test forward (TODO: set up optim here, run zero tests on pe grad)
    decoder.pos_embedding = nn.Parameter((torch.ones(MAX_LMX_SEQ_LEN, dtype=torch.float) + 500).unsqueeze(-1).repeat(1, hidden_dim))
    decoder.vocab_embedding.weight = nn.Parameter(torch.arange(VOCAB_LEN, dtype=torch.float).unsqueeze(-1).repeat(1, hidden_dim))
    print(f"Positional embedding grid\n{decoder.pos_embedding}\nEmbedding weights{decoder.vocab_embedding.weight}")

    lmx_seqs = [torch.tensor([0, 2, 3, 226]), torch.tensor([0, 2, 2, 3, 4, 226])]
    img_latent = torch.ones(2, 10, hidden_dim)
    latent_attention_mask = torch.cat([
        torch.cat([torch.tensor([False]).repeat(7), torch.tensor([True]).repeat(3)]).unsqueeze(0),
        torch.tensor([False]).repeat(10).unsqueeze(0)
    ], dim=0)
    print(f"Input lmx sequences\n{lmx_seqs}\nImage latent encoding\n{img_latent}\nLatent mask\n{latent_attention_mask}")
    pred = decoder(lmx_seqs, img_latent, latent_attention_mask)
    print(f"Decoder output prediction: {pred}")
    assert pred.shape == torch.Size([2, 6, hidden_dim])

    # test gradient flow

def test_vitomr():
    encoder = pretrained_debug_encoder
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt")
    debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
    vitomr = ViTOMR(encoder, debug_mae_state_dict, decoder)


    # test encoder is frozen (incl verify params are same before/after)
    
if __name__ == "__main__":
    test_decoder()