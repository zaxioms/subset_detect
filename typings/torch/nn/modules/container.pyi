"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict
from .module import Module
from torch._jit_internal import _copy_to_script_wrapper
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar, Union, overload

"""
This type stub file was generated by pyright.
"""
T = TypeVar('T')
class Container(Module):
    def __init__(self, **kwargs: Any) -> None:
        ...
    


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """
    @overload
    def __init__(self, *args: Module) -> None:
        ...
    
    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None:
        ...
    
    def __init__(self, *args: Any) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __getitem__(self: T, idx) -> T:
        ...
    
    def __setitem__(self, idx: int, module: Module) -> None:
        ...
    
    def __delitem__(self, idx: Union[slice, int]) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __len__(self) -> int:
        ...
    
    @_copy_to_script_wrapper
    def __dir__(self):
        ...
    
    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        ...
    
    def forward(self, input):
        ...
    


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """
    def __init__(self, modules: Optional[Iterable[Module]] = ...) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __getitem__(self, idx: int) -> Module:
        ...
    
    def __setitem__(self, idx: int, module: Module) -> None:
        ...
    
    def __delitem__(self, idx: Union[int, slice]) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __len__(self) -> int:
        ...
    
    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        ...
    
    def __iadd__(self: T, modules: Iterable[Module]) -> T:
        ...
    
    @_copy_to_script_wrapper
    def __dir__(self):
        ...
    
    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        ...
    
    def append(self: T, module: Module) -> T:
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        ...
    
    def extend(self: T, modules: Iterable[Module]) -> T:
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        ...
    
    def forward(self):
        ...
    


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged 
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to 
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """
    def __init__(self, modules: Optional[Mapping[str, Module]] = ...) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        ...
    
    def __setitem__(self, key: str, module: Module) -> None:
        ...
    
    def __delitem__(self, key: str) -> None:
        ...
    
    @_copy_to_script_wrapper
    def __len__(self) -> int:
        ...
    
    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        ...
    
    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        ...
    
    def clear(self) -> None:
        """Remove all items from the ModuleDict.
        """
        ...
    
    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        ...
    
    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        ...
    
    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        ...
    
    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values.
        """
        ...
    
    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        ...
    
    def forward(self):
        ...
    


class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be indexed like a regular Python
    list, but parameters it contains are properly registered, and will be
    visible by all :class:`~torch.nn.Module` methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """
    def __init__(self, parameters: Optional[Iterable[Parameter]] = ...) -> None:
        ...
    
    @overload
    def __getitem__(self, idx: int) -> Parameter:
        ...
    
    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...
    
    def __getitem__(self, idx):
        ...
    
    def __setitem__(self, idx: int, param: Parameter) -> None:
        ...
    
    def __setattr__(self, key: Any, value: Any) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __iter__(self) -> Iterator[Parameter]:
        ...
    
    def __iadd__(self: T, parameters: Iterable[Parameter]) -> T:
        ...
    
    def __dir__(self):
        ...
    
    def append(self: T, parameter: Parameter) -> T:
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        ...
    
    def extend(self: T, parameters: Iterable[Parameter]) -> T:
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        ...
    
    def extra_repr(self) -> str:
        ...
    
    def __call__(self, input):
        ...
    


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ParameterDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ParameterDict` (the argument to
      :meth:`~torch.nn.ParameterDict.update`).

    Note that :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter`) or an iterable of key-value pairs
            of type (string, :class:`~torch.nn.Parameter`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """
    def __init__(self, parameters: Optional[Mapping[str, Parameter]] = ...) -> None:
        ...
    
    def __getitem__(self, key: str) -> Parameter:
        ...
    
    def __setitem__(self, key: str, parameter: Parameter) -> None:
        ...
    
    def __delitem__(self, key: str) -> None:
        ...
    
    def __setattr__(self, key: Any, value: Any) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __iter__(self) -> Iterator[str]:
        ...
    
    def __contains__(self, key: str) -> bool:
        ...
    
    def clear(self) -> None:
        """Remove all items from the ParameterDict.
        """
        ...
    
    def pop(self, key: str) -> Parameter:
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        ...
    
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys.
        """
        ...
    
    def items(self) -> Iterable[Tuple[str, Parameter]]:
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        ...
    
    def values(self) -> Iterable[Parameter]:
        r"""Return an iterable of the ParameterDict values.
        """
        ...
    
    def update(self, parameters: Mapping[str, Parameter]) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        ...
    
    def extra_repr(self) -> str:
        ...
    
    def __call__(self, input):
        ...
    


