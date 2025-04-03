import matplotlib.pyplot as plt
import torch
import numpy as np

GRAND_STAFF_ROOT_DIR = "data/grandstaff-lmx.2024-02-12/grandstaff-lmx"
PRIMUS_PREPARED_ROOT_DIR = "data/primusPrepared"
DOREMI_PREPARED_ROOT_DIR = "data/doReMiPrepared"
OLIMPIC_SYNTHETIC_ROOT_DIR = "data/olimpic-1.0-synthetic.2024-02-12/olimpic-1.0-synthetic"
OLIMPIC_SCANNED_ROOT_DIR = "data/olimpic-1.0-scanned.2024-02-12/olimpic-1.0-scanned"

# dataset should be initialized with a ToTensor transformation for any images
def display_dataset_img(dataset, index): 
    data = dataset[index]

    if isinstance(data, tuple):
        img_tensors = []
        for item in data:
            if torch.is_tensor(item):
                img_tensors.append(item)

        rows = len(img_tensors)
        COLS = 1
        if rows == 0:
            print("No images retrieved to show. Make sure the dataset was initialized with a ToTensor transformation")
            return

        fig, axs = plt.subplots(rows, COLS)
        if rows == 1: # tuple contains one tensor, so one subplot and axs is just one axes object, not a list
            axs.imshow(img_tensors[0].permute(1, 2, 0).numpy())
        else:
            # one image per col
            for i in range(rows):
                ax = axs[i]
                ax.imshow(img_tensors[i].permute(1, 2, 0).numpy()) 
    else:
        fig, ax = plt.subplots()
        ax.imshow(data.permute(1, 2, 0).numpy())

    fig.suptitle(f"Index: {index}")
    plt.show()
    return data

def sample_pre_train_dataset(pre_train_dataset, num_samples, patch_size):
    sample_indices = np.floor(np.random.rand(num_samples) * len(pre_train_dataset)).astype(int)
    for i in range(num_samples):
        idx = sample_indices[i]
        input_img, _ = display_dataset_img(pre_train_dataset, idx)
        print(f"Input image shape: {input_img.shape}")
        print(f"Input image patch count: {(input_img.shape[1] / patch_size) * (input_img.shape[2] / patch_size) }")