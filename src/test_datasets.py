from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, PreTrainWrapper, OlimpicPreTrainWrapper, GrandStaffPreTrainWrapper
from utils import display_dataset_img, GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torchvision.transforms import ToTensor
import pytest

def test_grand_staff_dataset(mocker):
    # verify dataset retrieval works
    dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt")
    original, distorted, lmx = dataset[0]

    # verify transforms are being applied
    mock_transform = mocker.Mock()
    mock_transform.return_value = "transformed_img"
    dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", transform=mock_transform)
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
    for dataset in [GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt"),
                    OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt"),
                    OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt")]:
        print(f"Current dataset: {dataset}")
        assert transform(dataset[0][0]).shape[0] == 1
    
    for dataset in [PreparedDataset(PRIMUS_PREPARED_ROOT_DIR), PreparedDataset(DOREMI_PREPARED_ROOT_DIR)]:
        print(f"Current dataset: {dataset}")
        assert transform(dataset[0]).shape[0] == 1

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
    dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", transform=mock_transform)
    img, lmx = dataset[0]
    mock_transform.assert_called()
    assert img == "transformed_img"

# trying to parmeterize this test led to some weird (likely race condition) behavior with matplotlib
def test_to_tensor():
    dataset = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", transform=ToTensor())
    display_dataset_img(dataset, 0)

    dataset = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt", transform=ToTensor())
    display_dataset_img(dataset, 0)

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