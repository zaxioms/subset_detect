"""
This type stub file was generated by pyright.
"""

import sys

"""
This type stub file was generated by pyright.
"""
if sys.platform == 'win32' or sys.version_info < (3, 7):
    ...
else:
    ...
def register_after_fork(func):
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Arguments:
        func (function): Function taking no arguments to be called in the child after fork

    """
    ...
