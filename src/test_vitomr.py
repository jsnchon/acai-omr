import torch
from torch import nn
from models import OMREncoder, OMRDecoder, ViTOMR, NUM_CHANNELS
from pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH

debug_patch_size = 2
debug_encoder_kwargs = {"num_layers": 2, "num_heads": 2, "hidden_dim": 10, "mlp_dim": 100}
debug_encoder = OMREncoder(debug_patch_size, PE_MAX_HEIGHT, PE_MAX_WIDTH, **debug_encoder_kwargs)
 
def test_encoder_batchify():
    hidden_dim = NUM_CHANNELS * debug_patch_size ** 2
    encoder = OMREncoder(debug_patch_size, PE_MAX_HEIGHT, PE_MAX_WIDTH, hidden_dim=hidden_dim, num_heads=1)
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

def test_encoder_forward():
    encoder = debug_encoder
    x = [torch.rand(NUM_CHANNELS, 4, 4), torch.rand(NUM_CHANNELS, 4, 8)]
    x, mask = encoder(x)
    print(x)
    print(mask)
    assert x.shape == torch.Size([2, 8, debug_encoder_kwargs["hidden_dim"]])
    
if __name__ == "__main__":
    test_encoder_batchify()