import torch
from omr_train import EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR
from utils import cosine_anneal_with_warmup, plot_lr_schedule

def test_lr_scheduler():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    plot_lr_schedule(scheduler, optimizer, EPOCHS)

def test_prepare_lmx_seq():
    pass

if __name__ == "__main__":
    test_lr_scheduler()