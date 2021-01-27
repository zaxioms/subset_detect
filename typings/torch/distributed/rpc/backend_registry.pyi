"""
This type stub file was generated by pyright.
"""

import collections
import enum

BackendValue = collections.namedtuple("BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"])
_backend_type_doc = """
    An enum class of available backends.

    PyTorch ships with two builtin backends: ``BackendType.TENSORPIPE`` and
    ``BackendType.PROCESS_GROUP``. Additional ones can be registered using the
    :func:`~torch.distributed.rpc.backend_registry.register_backend` function.
"""
BackendType = enum.Enum(value="BackendType", names={  })
def backend_registered(backend_name):
    """
    Checks if backend_name is registered as an RPC backend.

    Arguments:
        backend_name (str): string to identify the RPC backend.
    Returns:
        True if the backend has been registered with ``register_backend``, else
        False.
    """
    ...

def register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler):
    """Registers a new RPC backend.

    Arguments:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
    ...

def construct_rpc_backend_options(backend, rpc_timeout=..., init_method=..., **kwargs):
    ...

def init_backend(backend, *args, **kwargs):
    ...

