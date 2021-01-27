"""
This type stub file was generated by pyright.
"""

import collections
import torch
from torch.nn import Module
from torch.jit._state import _enabled
from torch._six import with_metaclass

"""
This type stub file was generated by pyright.
"""
ScriptFunction = torch._C.ScriptFunction
if _enabled:
    Attribute = collections.namedtuple("Attribute", ["value", "type"])
else:
    def Attribute(value, type):
        ...
    
class OrderedDictWrapper(object):
    def __init__(self, _c) -> None:
        ...
    
    def keys(self):
        ...
    
    def values(self):
        ...
    
    def __len__(self):
        ...
    
    def __delitem__(self, k):
        ...
    
    def items(self):
        ...
    
    def __setitem__(self, k, v):
        ...
    
    def __contains__(self, k):
        ...
    
    def __getitem__(self, k):
        ...
    


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module, python_dict) -> None:
        ...
    
    def items(self):
        ...
    
    def __contains__(self, k):
        ...
    
    def __setitem__(self, k, v):
        ...
    
    def __getitem__(self, k):
        ...
    


class ScriptMeta(type):
    def __init__(cls, name, bases, attrs) -> None:
        ...
    


class _CachedForward(object):
    def __get__(self, obj, cls):
        ...
    


class ScriptWarning(Warning):
    ...


def script_method(fn):
    ...

class ConstMap:
    def __init__(self, const_mapping) -> None:
        ...
    
    def __getattr__(self, attr):
        ...
    


if _enabled:
    class ScriptModule(with_metaclass(ScriptMeta, Module)):
        """
        ``ScriptModule``s wrap a C++ ``torch::jit::Module``. ``ScriptModule``s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same as on a normal ``nn.Module``.
        """
        __jit_unused_properties__ = ...
        def __init__(self) -> None:
            ...
        
        forward = ...
        def __getattr__(self, attr):
            ...
        
        def __setattr__(self, attr, value):
            ...
        
        def define(self, src):
            ...
        
    
    
    class RecursiveScriptModule(ScriptModule):
        r"""
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions,  a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\s code to run without the need for a Python interpreter.

        ``ScriptModule``\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
        _disable_script_meta = ...
        def __init__(self, cpp_module) -> None:
            ...
        
        @property
        def graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. See :ref:`interpreting-graphs` for details.
            """
            ...
        
        @property
        def inlined_graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. This graph will be preprocessed to inline all function and method calls.
            See :ref:`interpreting-graphs` for details.
            """
            ...
        
        @property
        def code(self):
            r"""
            Returns a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See
            :ref:`inspecting-code` for details.
            """
            ...
        
        @property
        def code_with_constants(self):
            r"""
            Returns a tuple of:

            [0] a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `code`.
            [1] a ConstMap following the CONSTANT.cN format of the output in [0].
            The indices in the [0] output are keys to the underlying constant's values.

            See :ref:`inspecting-code` for details.
            """
            ...
        
        def save(self, *args, **kwargs):
            r"""
            save(f, _extra_files={})

            See :func:`torch.jit.save <torch.jit.save>` for details.
            """
            ...
        
        def save_to_buffer(self, *args, **kwargs):
            ...
        
        def get_debug_state(self, *args, **kwargs):
            ...
        
        def extra_repr(self):
            ...
        
        def graph_for(self, *args, **kwargs):
            ...
        
        @property
        def original_name(self):
            ...
        
        def define(self, src):
            ...
        
        def __getattr__(self, attr):
            ...
        
        def __setattr__(self, attr, value):
            ...
        
        def __getstate__(self):
            ...
        
        def __copy__(self):
            ...
        
        def __deepcopy__(self, memo):
            ...
        
        def forward_magic_method(self, method_name, *args, **kwargs):
            ...
        
        def __iter__(self):
            ...
        
        def __getitem__(self, idx):
            ...
        
        def __len__(self):
            ...
        
        def __contains__(self, key):
            ...
        
        def __dir__(self):
            ...
        
        def __bool__(self):
            ...
        
    
    
    _compiled_methods_allowlist = ("forward", "register_buffer", "register_parameter", "add_module", "_apply", "apply", "cuda", "cpu", "to", "type", "float", "double", "half", "state_dict", "_save_to_state_dict", "load_state_dict", "_load_from_state_dict", "_named_members", "parameters", "named_parameters", "buffers", "named_buffers", "children", "named_children", "modules", "named_modules", "zero_grad", "share_memory", "_get_name", "extra_repr", "_slow_forward", "_tracing_name", "eval", "train")
else:
    class ScriptModule(torch.nn.Module):
        def __init__(self, arg=...) -> None:
            ...
        
    
    
    class RecursiveScriptModule(ScriptModule):
        def __init__(self, arg=...) -> None:
            ...
        
    
    
def script(obj, optimize=..., _frames_up=..., _rcb=...):
    r"""
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    ``torch.jit.script`` can be used as a function for modules and functions, and as a decorator
    ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Arguments:
        obj (callable, class, or ``nn.Module``):  The ``nn.Module``, function, or class type to
                                                  compile.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFuncion

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super(MyModule, self).__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                    input = F.relu(self.conv1(input))
                    input = F.relu(self.conv2(input))
                    return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))
    """
    ...

def interface(obj):
    ...

class CompilationUnit(object):
    def __init__(self, lang=..., _frames_up=...) -> None:
        ...
    
    def define(self, lang, rcb=..., _frames_up=...):
        ...
    
    def __getattr__(self, attr):
        ...
    


