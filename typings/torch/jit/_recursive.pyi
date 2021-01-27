"""
This type stub file was generated by pyright.
"""

import torch
import collections
from typing import Dict, List, Set, Type
from torch.nn import Module

ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))
PropertyStub = collections.namedtuple('Property', ('resolution_callback', 'def_'))
ignored_attributes = ["_version", "_parameters", "_buffers", "_modules", "_initializing", "_backward_hooks", "_forward_hooks", "_forward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks", "dump_patches"]
def make_stub(func, name):
    ...

def make_stub_from_method(nn_module, method_name):
    ...

def make_stubs_from_exported_methods(mod):
    ...

_constant_types = (bool, float, int, str, type(None), torch.device, torch.layout, torch.dtype)
class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    def __init__(self, source, filename, file_lineno, leading_whitespace_len) -> None:
        ...
    


def infer_concrete_type_builder(nn_module, share_types=...):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module. This
    ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    ...

class ConcreteTypeStore(object):
    type_store: Dict[Type[Module], List[torch._C.ConcreteModuleType]]
    methods_compiled: Set[torch._C.ConcreteModuleType]
    def __init__(self) -> None:
        ...
    
    def get_or_create_concrete_type(self, nn_module):
        """
        Infer a ConcreteType from this `nn.Module` instance. Underlying JIT
        types are re-used if possible.
        """
        ...
    


concrete_type_store = ConcreteTypeStore()
def create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs):
    ...

def get_module_concrete_type(nn_module, share_types=...):
    """
    Gets a concrete type for nn_modules. If share_types is True, the concrete
    type is fetched from concrete_type_store. If it is False, a new concrete type
    is created without first searching concrete_type_store.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        share_types = Whether to share underlying JIT types between modules (if possible).

    Returns:
        A concrete type for nn_module.
    """
    ...

def create_script_module(nn_module, stubs_fn, share_types=...):
    """
    Creates a new ScriptModule from an nn.Module

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
    """
    ...

def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    ...

def script_model_defines_attr(script_model, attr):
    ...

def add_python_attr_to_scripted_model(script_model, orig, attr):
    ...

def get_overload_annotations(mod):
    ...

def get_overload_name_mapping(overload_info):
    ...

def make_stubs_for_overloads(overload_info):
    ...

def check_module_initialized(mod):
    ...

def infer_methods_to_compile(nn_module):
    """
    Implements the default rules for which methods should act as starting
    points for compilation (TODO add a link when the rules are published).
    """
    ...

def get_property_stubs(nn_module):
    """
    Create property stubs for the properties of the module by creating method
    stubs for the getter and setter.
    """
    ...

def interface_script(mod_interface, nn_module):
    """
    Makes a ScriptModule from an nn.Module, using the interface methods rule for
    determining which methods to compile.

    Arguments:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    ...

def try_compile_fn(fn, loc):
    ...

def wrap_cpp_module(cpp_module):
    """
    Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules
    """
    ...

def compile_unbound_method(concrete_type, fn):
    ...

def lazy_bind(concrete_type, unbound_method):
    """
    Returns a function that lazily binds `unbound_method` to a provided
    Module IValue, then invokes the method. We do this so that any Python
    shenanigans that will poison type sharing are impossible at compile
    time.
    """
    ...
