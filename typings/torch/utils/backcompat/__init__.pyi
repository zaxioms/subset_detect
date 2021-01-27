"""
This type stub file was generated by pyright.
"""

from torch._C import _get_backcompat_broadcast_warn, _get_backcompat_keepdim_warn, _set_backcompat_broadcast_warn, _set_backcompat_keepdim_warn

"""
This type stub file was generated by pyright.
"""
class Warning(object):
    def __init__(self, setter, getter) -> None:
        ...
    
    def set_enabled(self, value):
        ...
    
    def get_enabled(self):
        ...
    
    enabled = ...


broadcast_warning = Warning(_set_backcompat_broadcast_warn, _get_backcompat_broadcast_warn)
keepdim_warning = Warning(_set_backcompat_keepdim_warn, _get_backcompat_keepdim_warn)
