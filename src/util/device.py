import torch

"""
Use this when you reaaaly want to make sure pytorch is using your GPU.
"""


def fetch_device() -> str:
    """
    Returns 'cuda' if GPU is available otherwise 'cpu'.
    Can be used to determine where to move tensors to.
    :return: string 'cuda' or 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
