from datasets import GrandStaffLMXDataset, PreparedDataset
from PIL import Image
from utils import display_dataset_img
from torchvision.transforms import ToTensor
import pytest

def test_grand_staff_dataset(mocker):
    # verify dataset retrieval works
    dataset = GrandStaffLMXDataset("data/grandstaff-lmx.2024-02-12/grandstaff-lmx", "samples.train.txt")
    original, distorted, lmx = dataset[0]

    # verify transforms are being applied
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = GrandStaffLMXDataset("data/grandstaff-lmx.2024-02-12/grandstaff-lmx", "samples.train.txt", transform=mock_transform)
    original, distorted, lmx = dataset[0]
    mock_transform.assert_called()
    assert original == distorted == "transformed_img"

    # FIRST_EX_LMX_PATH was manually determined by reading the first id of the train split
    FIRST_EX_LMX_PATH = "data/grandstaff-lmx.2024-02-12/grandstaff-lmx/beethoven/piano-sonatas/sonata01-2/maj2_down_m-0-5.lmx"
    with open(FIRST_EX_LMX_PATH, "r") as f:
        first_ex_lmx = f.read()
    assert lmx == first_ex_lmx

    LAST_EX_LMX_PATH = "data/grandstaff-lmx.2024-02-12/grandstaff-lmx/scarlatti-d/keyboard-sonatas/L523K205/min3_up_m-96-100.lmx"
    original, distorted, lmx = dataset[-1]
    with open(LAST_EX_LMX_PATH, "r") as f:
        last_ex_lmx = f.read()
    assert lmx == last_ex_lmx

@pytest.mark.parametrize("root_dir", ["data/primusPrepared", "data/doReMiPrepared"])
def test_prepared_datasets(mocker, root_dir):
    # verify dataset retrieval works
    dataset = PreparedDataset(root_dir)
    img = dataset[0]

    # verify transforms are being applied
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = PreparedDataset(root_dir, transform=mock_transform)
    img = dataset[0]
    mock_transform.assert_called()
    assert img == "transformed_img"

# make sure that images are being converted into tensors correctly and are recoverable
def test_to_tensor():
    dataset = GrandStaffLMXDataset("data/grandstaff-lmx.2024-02-12/grandstaff-lmx", "samples.train.txt", transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = PreparedDataset("data/primusPrepared", transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = PreparedDataset("data/doReMiPrepared", transform=ToTensor())
    display_dataset_img(dataset, 0)