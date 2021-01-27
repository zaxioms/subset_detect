"""
This type stub file was generated by pyright.
"""

from torch.onnx.symbolic_helper import parse_args

def register_quantized_ops(domain, version):
    ...

def nchw2nhwc(g, input):
    ...

def nhwc2nchw(g, input):
    ...

def linear_prepack(g, weight, bias):
    ...

@parse_args('v', 'v', 'v', 'f', 'i')
def linear(g, input, weight, bias, scale, zero_point):
    ...

def conv_prepack(g, input, weight, bias, stride, padding, dilation, groups):
    ...

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    ...

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d_relu(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    ...

@parse_args('v', 'v', 'f', 'i')
def add(g, input_a, input_b, scale, zero_point):
    ...

@parse_args('v')
def relu(g, input):
    ...

@parse_args('v', 'f', 'i', 't')
def quantize_per_tensor(g, input, scale, zero_point, dtype):
    ...

@parse_args('v')
def dequantize(g, input):
    ...

def upsample_nearest2d(g, input, output_size, align_corners=..., scales_h=..., scales_w=...):
    ...

@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    ...

@parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=...):
    ...

def reshape(g, input, shape):
    ...

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, input, dim, start, end, step):
    ...

def cat(g, tensor_list, dim, scale=..., zero_point=...):
    ...

@parse_args('v')
def sigmoid(g, input):
    ...

