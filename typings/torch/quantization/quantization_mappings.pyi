"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
from torch import nn
from .stubs import DeQuantStub, QuantStub

"""
This type stub file was generated by pyright.
"""
STATIC_QUANT_MODULE_MAPPINGS = { nn.Linear: nnq.Linear,nn.ReLU: nnq.ReLU,nn.ReLU6: nnq.ReLU6,nn.Hardswish: nnq.Hardswish,nn.ELU: nnq.ELU,nn.Conv1d: nnq.Conv1d,nn.Conv2d: nnq.Conv2d,nn.Conv3d: nnq.Conv3d,nn.ConvTranspose1d: nnq.ConvTranspose1d,nn.ConvTranspose2d: nnq.ConvTranspose2d,nn.BatchNorm2d: nnq.BatchNorm2d,nn.BatchNorm3d: nnq.BatchNorm3d,nn.LayerNorm: nnq.LayerNorm,nn.GroupNorm: nnq.GroupNorm,nn.InstanceNorm1d: nnq.InstanceNorm1d,nn.InstanceNorm2d: nnq.InstanceNorm2d,nn.InstanceNorm3d: nnq.InstanceNorm3d,nn.Embedding: nnq.Embedding,nn.EmbeddingBag: nnq.EmbeddingBag,QuantStub: nnq.Quantize,DeQuantStub: nnq.DeQuantize,nnq.FloatFunctional: nnq.QFunctional,nni.ConvReLU1d: nniq.ConvReLU1d,nni.ConvReLU2d: nniq.ConvReLU2d,nni.ConvReLU3d: nniq.ConvReLU3d,nni.LinearReLU: nniq.LinearReLU,nni.BNReLU2d: nniq.BNReLU2d,nni.BNReLU3d: nniq.BNReLU3d,nniqat.ConvReLU2d: nniq.ConvReLU2d,nniqat.LinearReLU: nniq.LinearReLU,nniqat.ConvBn2d: nnq.Conv2d,nniqat.ConvBnReLU2d: nniq.ConvReLU2d,nnqat.Linear: nnq.Linear,nnqat.Conv2d: nnq.Conv2d }
QAT_MODULE_MAPPINGS = { nn.Linear: nnqat.Linear,nn.Conv2d: nnqat.Conv2d,nni.ConvBn2d: nniqat.ConvBn2d,nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,nni.ConvReLU2d: nniqat.ConvReLU2d,nni.LinearReLU: nniqat.LinearReLU }
DYNAMIC_QUANT_MODULE_MAPPINGS = { nn.Linear: nnqd.Linear,nn.LSTM: nnqd.LSTM,nn.LSTMCell: nnqd.LSTMCell,nn.RNNCell: nnqd.RNNCell,nn.GRUCell: nnqd.GRUCell }
_EXCLUDE_QCONFIG_PROPAGATE_LIST = DeQuantStub
_INCLUDE_QCONFIG_PROPAGATE_LIST = nn.Sequential
FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS = { F.elu: torch._ops.ops.quantized.elu,F.hardswish: torch._ops.ops.quantized.hardswish,F.instance_norm: torch._ops.ops.quantized.instance_norm,F.layer_norm: torch._ops.ops.quantized.layer_norm }
def register_static_quant_module_mapping(float_source_module_class, static_quant_target_module_class):
    ''' Register a mapping from `float_source__module_class` to `static_quant_target_module_class`
    `static_quant_target_module_class` must have from_float defined as a class method
    The mapping is used in the convert step of post training static quantization to
    convert a float module to a statically quantized module.
    '''
    ...

def get_static_quant_module_mappings():
    ''' Get module mapping for post training static quantization
    '''
    ...

def get_static_quant_module_class(float_module_class):
    ''' Get the statically quantized module class corresponding to
    the floating point module class
    '''
    ...

def register_qat_module_mapping(float_source_module_class, qat_target_module_class):
    '''Register a mapping from `float_source_module_class` to `qat_target_module_class`,
    `qat_target_module_class` must have from_float defined as a class method
    This mapping is used in prepare step of quantization aware training to swap
    a float module to a qat module.
    '''
    ...

def get_qat_module_mappings():
    ''' Get module mapping for quantization aware training
    '''
    ...

def register_dynamic_quant_module_class(float_source_module_class, dynamic_quant_target_module_class):
    ''' Register a mapping from `float_source_module_class` to `dynamic_quant_target_module_class`,
    `dynamic_quant_target_module_class` must have from_float defined as a class method
    This mapping is used in convert step of post training dynamic
    quantization to swap a float module to a dynamically quantized
    module.
    '''
    ...

def get_dynamic_quant_module_mappings():
    ''' Get module mapping for post training dynamic quantization
    '''
    ...

def get_qconfig_propagation_list():
    ''' Get the list of module types that we'll attach qconfig
    attribute to in prepare
    '''
    ...

def get_compare_output_module_list():
    ''' Get list of module class types that we will record output
    in numeric suite
    '''
    ...

def register_quantized_operator_mapping(float_op, quantized_op):
    ''' Register a mapping from `floating_point_op` (torch or functional) to `quantized_op`
    This is used in convert step of fx based graph mode quantization
    to convert a float op to quantized op.
    '''
    ...

def get_quantized_operator(float_op):
    ''' Get the quantized operator corresponding to the float operator
    '''
    ...
