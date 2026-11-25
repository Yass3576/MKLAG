import numpy as np
import torch

from torch.nn import Module


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x

def load_weights(model: Module, ckpt: str, strict: bool=True, device: str='cuda') -> None:
    """
    @brief
    Load pretrained weights from a checkpoint file into the model.

    @param model (torch.nn.Module)
        The model into which weights will be loaded.

    @param ckpt (str)
        Path to the checkpoint file containing pretrained weights.

    @return None
    """
    state_dict = torch.load(ckpt, map_location=device)["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if ".fa2s_layer." not in k}
    model.load_state_dict(state_dict, strict=strict)
    print('Weights loaded')


def get_device():
    """
    @brief Get the available device.

    @details Checks if CUDA is available and returns 'cuda' or 'cpu' accordingly.

    @return device (string) String representing the device name.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def flatten(xss: list) -> list:
    """
    @brief Flatten a list of lists.

    @param xss (list) List of lists to flatten.

    @return (list) Flattened list.
    """
    return [x for xs in xss for x in xs]


def head_classes(ckpt, ordinal=False):
    state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
    n_out = [v for k, v in state_dict.items() if "scene_head" in k and v.dim() == 2][-1].shape[0]
    return n_out + 1 if ordinal else n_out