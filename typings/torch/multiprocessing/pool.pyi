"""
This type stub file was generated by pyright.
"""

import multiprocessing

def clean_worker(*args, **kwargs):
    ...

class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.
    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data."""
    ...


