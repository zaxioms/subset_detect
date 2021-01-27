"""
This type stub file was generated by pyright.
"""

import sys
import ast
import string
from torch._C._jit_tree_views import Assert, Break, ClassDef, Continue, Delete, For, Raise, With

_reserved_prefix = '__jit'
_reserved_names = 'print'
_identifier_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)
def is_reserved_name(name):
    ...

pretty_node_names = { ast.FunctionDef: "function definitions",ast.For: "for loops",ast.Delete: "del statements",ast.ClassDef: "class definitions",ast.With: "with statements",ast.Raise: "raise statements",ast.Assert: "assertions",ast.Import: "import statements",ast.ImportFrom: "import statements",ast.Global: "global variables",ast.Break: "break statements",ast.Continue: "continue statements" }
node_start_tokens = { ast.FunctionDef: "def",ast.For: "for",ast.Delete: "del",ast.ClassDef: "class",ast.With: "with",ast.Raise: "raise",ast.Assert: "assert",ast.Import: "import",ast.ImportFrom: "from",ast.Global: "global",ast.Break: "break",ast.Continue: "continue" }
if sys.version_info >= (3, 6):
    ...
class FrontendError(Exception):
    def __init__(self, source_range, msg) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class NotSupportedError(FrontendError):
    ...


class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason=...) -> None:
        ...
    


class FrontendTypeError(FrontendError):
    ...


def build_withitems(ctx, items):
    ...

def build_stmts(ctx, stmts):
    ...

def get_class_properties(cls, self_name):
    """
    Get a list of Property objects representing the properties of a class.

    Arguments:
        cls:  The class to get properties of.
        self_name: The name of the class that the properties should belong to.
    Returns:
        A list of Property objects corresponding to the properties of cls. Property
        here refers to the subclass of TreeView.
    """
    ...

def get_jit_class_def(cls, self_name):
    ...

def get_jit_def(fn, def_name, self_name=...):
    """
    Build a JIT AST (TreeView) from the given function.

    Arguments:
        fn: A function object to compile
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    """
    ...

class Builder(object):
    def __call__(self, ctx, node):
        ...
    


def build_class_def(ctx, py_def, methods, properties, self_name):
    ...

def build_def(ctx, py_def, type_line, def_name, self_name=...):
    ...

_vararg_kwarg_err = "Compiled functions can't take variable number of arguments " "or use keyword-only arguments with defaults"
def build_param_list(ctx, py_args, self_name):
    ...

def build_param(ctx, py_arg, self_name, kwarg_only):
    ...

def get_default_args(fn):
    ...

def get_default_args_for_class(cls):
    """
    Get default arguments for all methods in a class (except for static methods).

    Args:
        cls: type - The class type to inspect for default arguments.
    Returns:
        A Dict[str, Dict[str, Any]] which maps each method name to a Dict[str, Any]
        that maps each argument name to its default value.
    """
    ...

class WithItemBuilder(Builder):
    @staticmethod
    def build_withitem(ctx, item):
        ...
    


class StmtBuilder(Builder):
    augassign_map = ...
    @staticmethod
    def build_Expr(ctx, stmt):
        ...
    
    @staticmethod
    def build_Assign(ctx, stmt):
        ...
    
    @staticmethod
    def build_AnnAssign(ctx, stmt):
        ...
    
    @staticmethod
    def build_Delete(ctx, stmt):
        ...
    
    @staticmethod
    def build_Return(ctx, stmt):
        ...
    
    @staticmethod
    def build_Raise(ctx, stmt):
        ...
    
    @staticmethod
    def build_Assert(ctx, stmt):
        ...
    
    @staticmethod
    def build_AugAssign(ctx, stmt):
        ...
    
    @staticmethod
    def build_While(ctx, stmt):
        ...
    
    @staticmethod
    def build_For(ctx, stmt):
        ...
    
    @staticmethod
    def build_If(ctx, stmt):
        ...
    
    @staticmethod
    def build_Print(ctx, stmt):
        ...
    
    @staticmethod
    def build_Pass(ctx, stmt):
        ...
    
    @staticmethod
    def build_Break(ctx, stmt):
        ...
    
    @staticmethod
    def build_Continue(ctx, stmt):
        ...
    
    @staticmethod
    def build_With(ctx, stmt):
        ...
    


class ExprBuilder(Builder):
    binop_map = ...
    unop_map = ...
    boolop_map = ...
    cmpop_map = ...
    @staticmethod
    def build_Attribute(ctx, expr):
        ...
    
    @staticmethod
    def build_Call(ctx, expr):
        ...
    
    @staticmethod
    def build_Ellipsis(ctx, expr):
        ...
    
    @staticmethod
    def build_Name(ctx, expr):
        ...
    
    @staticmethod
    def build_NameConstant(ctx, expr):
        ...
    
    @staticmethod
    def build_BinOp(ctx, expr):
        ...
    
    @staticmethod
    def build_UnaryOp(ctx, expr):
        ...
    
    @staticmethod
    def build_BoolOp(ctx, expr):
        ...
    
    @staticmethod
    def build_IfExp(ctx, expr):
        ...
    
    @staticmethod
    def build_Compare(ctx, expr):
        ...
    
    @staticmethod
    def build_Subscript(ctx, expr):
        ...
    
    @staticmethod
    def build_List(ctx, expr):
        ...
    
    @staticmethod
    def build_Tuple(ctx, expr):
        ...
    
    @staticmethod
    def build_Dict(ctx, expr):
        ...
    
    @staticmethod
    def build_Num(ctx, expr):
        ...
    
    @staticmethod
    def build_Constant(ctx, expr):
        ...
    
    @staticmethod
    def build_Str(ctx, expr):
        ...
    
    @staticmethod
    def build_JoinedStr(ctx, expr):
        ...
    
    @staticmethod
    def build_ListComp(ctx, stmt):
        ...
    
    @staticmethod
    def build_Starred(ctx, expr):
        ...
    


build_expr = ExprBuilder()
build_stmt = StmtBuilder()
build_withitem = WithItemBuilder()
def find_before(ctx, pos, substr, offsets=...):
    ...

