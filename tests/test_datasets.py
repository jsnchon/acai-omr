from acai_omr.train.datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, PreTrainWrapper, OlimpicPreTrainWrapper, GrandStaffPreTrainWrapper, GrandStaffOMRTrainWrapper
from acai_omr.utils.utils import display_dataset_img, DynamicResize
from acai_omr.config import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torchvision.transforms import ToTensor
import pytest
import torch
import matplotlib.pyplot as plt

def test_grand_staff_dataset(mocker):
    # verify dataset retrieval works
    dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt")
    original, distorted, lmx = dataset[0]

    # verify transforms are being applied
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=mock_transform)
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

def test_grayscale():
    transform = ToTensor()
    for name, dataset in {"grandstaff": GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt"),
                    "olimpic_scanned": OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt"),
                    "olimpic_synthetic": OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt")}.items():
        print(f"Current dataset: {dataset}")
        img_t = transform(dataset[0][0])
        assert img_t.shape[0] == 1
        plt.imshow(img_t.squeeze(0), cmap="gray")
        plt.savefig(f"{name}.png")
    
    for name, dataset in {"primus": PreparedDataset(PRIMUS_PREPARED_ROOT_DIR), 
                          "doremi": PreparedDataset(DOREMI_PREPARED_ROOT_DIR)}.items():
        print(f"Current dataset: {dataset}")
        img_t = transform(dataset[0])
        assert img_t.shape[0] == 1
        plt.imshow(img_t.squeeze(0), cmap="gray")
        plt.savefig(f"{name}.png")

@pytest.mark.parametrize("root_dir", [PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR])
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

# only the synthetic olimpic dataset has a train split
def test_olimpic_datasets(mocker):
    dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt")
    img, lmx = dataset[0]

    dataset = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt")
    img, lmx = dataset[0]

    # verify transforms are being applied
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=mock_transform)
    img, lmx = dataset[0]
    mock_transform.assert_called()
    assert img == "transformed_img"

# # trying to parmeterize this test led to some weird (likely race condition) behavior with matplotlib
# def test_to_tensor():
#     dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=ToTensor())
#     display_dataset_img(dataset, 0)
# 
#     dataset = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=ToTensor())
#     display_dataset_img(dataset, 0)
# 
#     dataset = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=ToTensor())
#     display_dataset_img(dataset, 0)
# 
#     dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=ToTensor())
#     display_dataset_img(dataset, 0)
# 
#     dataset = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt", img_transform=ToTensor())
#     display_dataset_img(dataset, 0)

test_params = [
    (PreTrainWrapper, PreparedDataset, (PRIMUS_PREPARED_ROOT_DIR,)),
    (PreTrainWrapper, PreparedDataset, (DOREMI_PREPARED_ROOT_DIR,)),
    (OlimpicPreTrainWrapper, OlimpicDataset, (OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt")),
    (OlimpicPreTrainWrapper, OlimpicDataset, (OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt")),
]
@pytest.mark.parametrize("wrapper_class, base_class, base_class_args", test_params)
def test_pre_train_wrappers(mocker, wrapper_class, base_class, base_class_args):
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = wrapper_class(base_class(*base_class_args), transform=mock_transform)
    
    input_img, target_img = dataset[0]
    mock_transform.assert_called()
    assert input_img == "transformed_img" and target_img != "transformed_img"

# GrandStaff wrapper is a special case since its constructor takes additional arguments beyond the other wrappers
def test_grand_staff_pre_train_wrapper(mocker):
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    base_dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt")
    dataset = GrandStaffPreTrainWrapper(base_dataset, 1, mock_transform)

    input_img, target_img = dataset[0]
    mock_transform.assert_called()
    assert input_img == "transformed_img" and target_img != "transformed_img"

def test_dynamic_resize():
    patch_size = 2
    max_seq_len = 10
    pe_max_height = 4
    pe_max_width = 8 
    resize = DynamicResize(patch_size, max_seq_len, pe_max_height, pe_max_width, False)

    img = resize(torch.rand(1, 6, 10))
    assert (img.shape[-1] / patch_size) * (img.shape[-2] / patch_size) <= 10

    img = resize(torch.rand(1, 10, 6))
    assert (img.shape[-1] / patch_size) * (img.shape[-2] / patch_size) <= 10

    img = resize(torch.rand(1, 100, 200))
    assert img.shape[-1] / patch_size < pe_max_width and img.shape[-2] / patch_size < pe_max_height

def test_grand_staff_omr_train_wrapper(mocker):
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    base_dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt")
    dataset = GrandStaffOMRTrainWrapper(base_dataset, 1, mock_transform)

    input_img, lmx = dataset[0]
    mock_transform.assert_called()
    assert input_img == "transformed_img" and lmx != "transformed_img"
    assert type(lmx) == str

    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = GrandStaffOMRTrainWrapper(base_dataset, 0, mock_transform)
    input_img, lmx = dataset[0]
    mock_transform.assert_not_called()
    assert input_img != "transformed_img" and lmx != "transformed_img"
    assert type(lmx) == str
