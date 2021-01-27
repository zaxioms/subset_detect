"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict

FUSION_PATTERNS = OrderedDict()
def register_fusion_pattern(pattern):
    ...

def get_fusion_patterns():
    ...

QUANTIZATION_PATTERNS = OrderedDict()
def register_quant_pattern(pattern):
    ...

def get_quant_patterns():
    ...

DYNAMIC_QUANTIZATION_PATTERNS = OrderedDict()
def register_dynamic_quant_pattern(pattern):
    ...

def get_dynamic_quant_patterns():
    ...

def is_match(modules, node, pattern, max_uses=...):
    """ Matches a node in fx against a pattern
    """
    ...

