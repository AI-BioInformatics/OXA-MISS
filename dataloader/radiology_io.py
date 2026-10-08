"""Reading the unpooled CT / MRI encoder outputs (ccRCC/features/<encoder>/<CT|MR>/<TUMOR>/<case>.pt, shape
(1, N regions, dim)): the token matrices of the radiology encoder of OXA_MISS_final.

The .pt files are MONAI MetaTensors saved with a development MONAI version (metadata key "original_pixdim",
absent from every released MONAI), so torch.load fails even with monai installed. The tensor values do not
depend on MONAI: the unpickler below maps the MONAI classes to a plain tensor and drops the metadata (affine /
spacing, not used). Checked: the mean over the tokens equals the mean-pooled vectors of features_meanpooled."""
import pickle
import types

import numpy as np
import torch


class _PlainTensor(torch.Tensor):
    """Stand-in for monai MetaTensor: keeps the values, ignores the metadata."""
    def __setstate__(self, state):
        pass


class _MonaiFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("monai"):
            return _PlainTensor if name == "MetaTensor" else (lambda *args, **kwargs: None)
        return super().find_class(module, name)


_PICKLE_MODULE = types.SimpleNamespace(Unpickler=_MonaiFreeUnpickler, load=pickle.load, __name__="monai_free_pickle")


def load_radiology_tokens(path):
    """(N, dim) float32 tensor of an unpooled feature file (.pt MetaTensor, .npy or .npz)."""
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PICKLE_MODULE)
        tokens = torch.Tensor(data).clone() if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data))
    else:
        data = np.load(path, allow_pickle=True)
        tokens = torch.as_tensor(np.asarray(data[list(data.keys())[0]] if path.endswith(".npz") else data))
    tokens = tokens.float()
    return tokens.reshape(-1, tokens.shape[-1])   # (1, N, dim) -> (N, dim)
