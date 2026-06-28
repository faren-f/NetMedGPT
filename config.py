import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker so data loading is deterministic."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_reproducibility_settings(seed: int) -> None:
    """Print all reproducibility settings at startup."""
    print("=" * 50)
    print("Reproducibility settings")
    print(f"  seed                     : {seed}")
    print(f"  torch.manual_seed        : {seed}")
    print(f"  torch.cuda.manual_seed   : {seed}")
    print(f"  numpy / random seed      : {seed}")
    print(f"  cudnn.deterministic      : {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark          : {torch.backends.cudnn.benchmark}")
    print("=" * 50)
