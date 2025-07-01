import torch
from models import Encoder

def test_encoder_batchify():
    encoder = Encoder(patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    batch = encoder.batchify(x)
    print(batch)
    assert torch.equal(batch.offsets(), torch.tensor([0, 4, 12]))

if __name__ == "__main__":
    test_encoder_batchify()