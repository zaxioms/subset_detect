"""
This type stub file was generated by pyright.
"""

import contextlib

"""
This type stub file was generated by pyright.
"""
DEFAULT_FLAGS = ["gpustarttimestamp", "gpuendtimestamp", "gridsize3d", "threadblocksize", "streamid", "enableonstart 0", "conckerneltrace"]
def init(output_file, flags=..., output_mode=...):
    ...

def start():
    ...

def stop():
    ...

@contextlib.contextmanager
def profile():
    ...
