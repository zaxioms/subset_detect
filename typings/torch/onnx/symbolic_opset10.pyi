"""
This type stub file was generated by pyright.
"""

from torch.nn.modules.utils import _pair, _single, _triple
from torch.onnx.symbolic_helper import parse_args

@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=...):
    ...

@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=...):
    ...

max_pool1d = _max_pool("max_pool1d", _single, 1, return_indices=False)
max_pool2d = _max_pool("max_pool2d", _pair, 2, return_indices=False)
max_pool3d = _max_pool("max_pool3d", _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool("max_pool1d_with_indices", _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool("max_pool2d_with_indices", _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool("max_pool3d_with_indices", _triple, 3, return_indices=True)
avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)
upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")
def slice(g, self, *args):
    ...

@parse_args('v', 'is')
def flip(g, input, dims):
    ...

def fmod(g, input, other):
    ...

@parse_args('v', 'v', 'v', 'i', 'i', 'i', 'v', 'i')
def embedding_bag(g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    ...

@parse_args('v', 't', 'i', 'i', 'i')
def fake_quantize_per_tensor_affine(g, inputs, scale, zero_point, quant_min=..., quant_max=...):
    ...

