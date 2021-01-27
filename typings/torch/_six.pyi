"""
This type stub file was generated by pyright.
"""

import builtins
import io
import math
import sys

"""
This type stub file was generated by pyright.
"""
inf = math.inf
nan = math.nan
string_classes = (str, bytes)
int_classes = int
FileNotFoundError = builtins.FileNotFoundError
StringIO = io.StringIO
container_abcs = collections.abc
PY3 = sys.version_info[0] == 3
PY37 = sys.version_info[0] == 3 and sys.version_info[1] >= 7
def with_metaclass(meta: type, *bases) -> type:
    """Create a base class with a metaclass."""
    class metaclass(meta):
        ...
    
    

def get_function_from_type(cls, name):
    ...

def istuple(obj) -> bool:
    ...

def bind_method(fn, obj, obj_type):
    ...
