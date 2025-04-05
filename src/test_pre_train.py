from pre_train import BucketBatchSampler
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
    # dataloader = DataLoader(dataset, batch_sampler=sampler) can't do dataloader without padding collate fn
    # for i, batch in enumerate(dataloader):
    for i, indices in enumerate(sampler):
        print(f"Batch {i + 1}: indices returned by sampler: {indices}")
        #print(f"Batch {i}: {batch} Mock image tensor shape: {batch[0].shape}")
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