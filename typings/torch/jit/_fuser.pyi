"""
This type stub file was generated by pyright.
"""

import contextlib
import torch

"""
This type stub file was generated by pyright.
"""
@contextlib.contextmanager
def optimized_execution(should_optimize):
    """
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    """
    ...

@contextlib.contextmanager
def fuser(name):
    """
    A context manager that facilitates switching between
    backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    """
    ...

last_executed_optimized_graph = torch._C._last_executed_optimized_graph
