import torch
from models import Encoder

def test_encoder_batchify():
    encoder = Encoder(patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    batch, mask = encoder.batchify(x)
    print(batch)
    print(mask)
    # assert torch.equal(batch.offsets(), torch.tensor([0, 4, 12]))
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

if __name__ == "__main__":
    test_encoder_forward()