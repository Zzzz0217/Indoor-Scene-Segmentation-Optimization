# test if cuda is available
import torch

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
    exit(1)
    