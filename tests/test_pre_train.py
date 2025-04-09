from src.utils import BucketBatchSampler
from src.models import AdaptivePadPatchEmbed, MaskedAttention
from src.pre_train import pre_train_collate_fn
from torch.utils.data import Dataset, DataLoader
import torch
import pytest
import logging

class MockDataset(Dataset):
    def __init__(self, mock_tensors):
        self.mock_tensors = mock_tensors

    def __len__(self):
        return len(self.mock_tensors)

    def __getitem__(self, idx):
        return self.mock_tensors[idx], "some other data"

# test cases where some buckets are left empty, some need to go into leftover, shuffling vs not, batching
test_args = [
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 100), torch.rand(600, 600)], 1, False),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(40, 600), torch.rand(40, 300)], 1, False),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 256), torch.rand(60, 40)], 1, False),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 100), torch.rand(600, 600)], 1, True),
    ([torch.rand(300, 300), torch.rand(600, 600)], 1, True),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 256), torch.rand(60, 40)], 1, True),
    ([torch.rand(10, 10), torch.rand(5, 25), torch.rand(3, 15), torch.rand(20, 2), torch.rand(3, 3),
      torch.rand(256, 256), torch.rand(300, 301), torch.rand(300, 300), torch.rand(100, 100), torch.rand(600, 600)], 2, False),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 256), torch.rand(60, 40)], 2, False),
    ([torch.rand(10, 10), torch.rand(5, 25), torch.rand(3, 15), torch.rand(20, 2), torch.rand(3, 3),
      torch.rand(256, 256), torch.rand(300, 301), torch.rand(300, 300), torch.rand(100, 100), torch.rand(600, 600)], 2, True),
    ([torch.rand(10, 10), torch.rand(300, 300), torch.rand(100, 256), torch.rand(60, 40)], 2, True),
    ]

@pytest.mark.parametrize("mock_data, batch_size, shuffle", test_args)
def test_bucket_batch_sampler(caplog, mock_data, batch_size, shuffle):
    caplog.set_level(logging.DEBUG, logger="pre_train")
    bucket_boundaries = [(256, 400), (512, 512)]
    dataset = MockDataset(mock_data)
    sampler = BucketBatchSampler(dataset, bucket_boundaries, batch_size, shuffle)
    assert len(sampler.buckets) > 0
    for i, indices in enumerate(sampler):
        print(f"Batch {i + 1}: indices returned by sampler: {indices}")
    print(caplog.text)

# tests in main to use if Pytest stdout capturing is being weird
dataset = MockDataset(test_args[7][0])
sampler = BucketBatchSampler(dataset, [(256, 400), (512, 512)], 2, True)
for i, indices in enumerate(sampler):
    print(f"Batch {i + 1}: indices returned by sampler: {indices}")
dataset = MockDataset(test_args[-1][0])
sampler = BucketBatchSampler(dataset, [(256, 400), (512, 512)], 2, True)
for i, indices in enumerate(sampler):
    print(f"Batch {i + 1}: indices returned by sampler: {indices}")

test_args = [
    ([torch.rand(3, 4, 4), torch.rand(3, 4, 6)]),
    ([torch.rand(3, 4, 4), torch.rand(3, 4, 4)]),
    ([torch.rand(3, 4, 4), torch.rand(3, 4, 6), torch.rand(3, 2, 4), torch.rand(3, 2, 6)]),
    ([torch.rand(3, 4, 6), torch.rand(3, 4, 4), torch.rand(3, 2, 4), torch.rand(3, 2, 6), torch.rand(3, 10, 10)]),
    ([torch.rand(3, 4, 4), torch.rand(3, 4, 6), torch.rand(3, 2, 4), torch.rand(3, 2, 6), torch.rand(3, 10, 10), torch.rand(3, 10, 12), torch.rand(3, 12, 12)]),
    ]
@pytest.mark.parametrize("mock_data", test_args)
def test_embedding(mock_data):
    bucket_boundaries = [(2, 3), (8, 8)]
    dataset = MockDataset(mock_data)
    sampler = BucketBatchSampler(dataset, bucket_boundaries, 2, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pre_train_collate_fn) 
    embed = AdaptivePadPatchEmbed(2, 6)
    for i, batch in enumerate(dataloader):
        embeddings, mask = embed(batch[0])
        print(f"Batch {i + 1} embedding tensor: {embeddings}")
        print(f"Batch tensor shape: {embeddings.shape}")
        print(f"Mask tensor: {mask}")

# just have to add print statements inside the module class to debug (then remove them for efficiency)
@pytest.mark.parametrize("mock_data", test_args)
def test_attention_masking(mock_data):
    bucket_boundaries = [(2, 3), (8, 8)]
    dataset = MockDataset(mock_data)
    sampler = BucketBatchSampler(dataset, bucket_boundaries, 2, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pre_train_collate_fn) 
    embed = AdaptivePadPatchEmbed(2, 6)
    for i, batch in enumerate(dataloader):
        embeddings, mask = embed(batch[0])
        print(f"Batch {i + 1} embedding tensor shape: {embeddings.shape}")
        print(f"Batch mask tensor: {mask}")
        attention_layer = MaskedAttention(6, 1)
        embeddings = attention_layer(embeddings, mask) # test with multiple heads too