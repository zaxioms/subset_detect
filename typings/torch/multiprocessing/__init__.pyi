"""
This type stub file was generated by pyright.
"""

import torch
import sys
import multiprocessing
from .reductions import init_reductions
from multiprocessing import *
from .spawn import ProcessContext, SpawnContext, _supports_context, spawn, start_processes

"""
This type stub file was generated by pyright.
"""
if sys.version_info < (3, 3):
    ...
if sys.platform == 'darwin' or sys.platform == 'win32':
    ...
else:
    _sharing_strategy = 'file_descriptor'
    _all_sharing_strategies = ('file_descriptor', 'file_system')
def set_sharing_strategy(new_strategy):
    """Sets the strategy for sharing CPU tensors.

    Arguments:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    ...

def get_sharing_strategy():
    """Returns the current strategy for sharing CPU tensors."""
    ...

def get_all_sharing_strategies():
    """Returns a set of sharing strategies supported on a current system."""
    ...

