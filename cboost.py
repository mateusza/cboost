#!/usr/bin/env python3.10

import inspect
import ast
import dis
import os
import hashlib
import ctypes

_boosted_functions: dict = {}
_boosted_py_src: str = ''
_disabled = False

def disable():
    global _disabled
    _disabled = True

class CppSource:
    includes: set
    declarations: list
    definitions: list

    def __str__(self):
        includes = [f'#include <{inc}>' for inc in self.includes]
        declarations = [f'{dec};' for dec in self.decarations]
        definitions = [f'{def_}\n' for def_ in self.definitions]
        return '\n'.join([*includes, *declarations, *definitions])

class AST:
    """
    cboost Abstract Syntax Tree element
    """
    def _render(self, indent: int = 4, curr_indent: str='', next_indent: str='', **kwargs):
        """
        Render element into C/C++ source code
        """
        return f'{curr_indent}/* AST {self} */'

    @classmethod
    def _from_py(cls, ast_element: ast.AST):
        """
        Create new element by converting Python AST Element
        """
        print(f'{ast_element=}')
        print(f'{type(ast_element)=}')
        print(f'{ast.dump(ast_element, indent=8)}')  
        raise Exception(f'Not implemented {cls=}')

class arg(AST):
    arg: str
    annotation: AST

    def __init__(self, arg: str = "", annotation: AST = None):
        self.arg = arg
        self.annotation = annotation

    @classmethod
    def _from_py(cls, a: ast.arg):
        """
        arg
        """
        return cls(**{
            'arg': a.arg,
            'annotation': convert(a.annotation)
        })

    def _render(self, **kwargs):
        try:
            return_type = convert_type(self.annotation.id)
        except AttributeError:
            raise Exception('Arguments without type annotations are not supported')
        return return_type + ' ' + self.arg

class arguments(AST):
    args: list
    posonlyargs: list
    kwonlyargs: list
    kw_defaults: list
    defaults: list

    def __init__(self, args, posonlyargs, kwonlyargs, kw_defaults, defaults):
        self.args = args
        self.posonlyargs = posonlyargs
        self.kwonlyargs = kwonlyargs
        self.kw_defaults = kw_defaults
        self.defaults = defaults

    @classmethod
    def _from_py(cls, a: ast.arguments):
        return cls(**{
            'args': convert_list(a.args),
            'posonlyargs': convert_list(a.posonlyargs),
            'kwonlyargs': convert_list(a.kwonlyargs),
            'kw_defaults': convert_list(a.kw_defaults),
            'defaults': convert_list(a.defaults)
        })

    def _render(self, **kwargs):
        # FIXME: defaults

        return ', '.join([
            render(a) for na, a in enumerate(self.args)
        ])

class boolop(AST):
    """
    cboost AST boolean op
    """

    @classmethod
    def _from_py(cls, the_op: ast.boolop):
        return cls()

    def _render(self, **kwargs):
        return self.r


class cmpop(AST):
    """
    cboost AST comparing operation
    """

    @classmethod
    def _from_py(cls, the_op: ast.cmpop):
        return cls()

    def _render(self, **kwargs):
        return self.r

class expr(AST):
    """
    cboost AST expression
    """

class mod(AST):
    """
    cboost AST mod
    """

class operator(AST):
    """
    cboost AST Operator object
    """

    @classmethod
    def _from_py(cls, the_op: ast.operator):
        return cls()

    def _render(self, **kwargs):
        return self.r

class stmt(AST):
    """
    cboost AST statement
    """

class Add(operator):
    """
    cboost Add
    """
    r: str = '+'

class And(boolop):
    r: str = '&&'

class AnnAssign(stmt):
    """
    cboost AnnAssign
    """
    target: object = None
    annotation: object = None
    value: object = None
    simple: int = 0

    def __init__(self, target: AST = None, annotation: AST = None, value: expr=None, simple: int =0):
        self.target = target
        self.value = value
        self.annotation = annotation
        self.simple = simple

    @classmethod
    def _from_py(cls, annassign: ast.AnnAssign):
        """
        convert
        """
        target = convert(annassign.target)
        value = convert(annassign.value)
        annotation = convert(annassign.annotation)
        simple = annassign.simple
        a = cls(target=target, annotation=annotation, value=value, simple=simple)
        return a

    def _render(self, curr_indent: str = '', semicolon = True, **kwargs):
        return curr_indent + render(self.annotation) + ' ' + render(self.target) + ' = ' + render(self.value, brackets=False) + ';' if semicolon else ''


class Assign(stmt):
    """
    cboost Assign
    """
    targets: list = []
    value = None
    def __init__(self, targets=[], value=None):
        self.targets = targets
        self.value = value

    @classmethod
    def _from_py(cls, assign: ast.Assign):
        """
        Convert Python AST Assign
        """
        # TODO: Multiple assignments
        targets = convert_list(assign.targets)
        value = convert(assign.value)
        a = cls(targets=targets, value=value)
        return a

    def _render(self, curr_indent: str = '', semicolon = True, **kwargs):
        return curr_indent + render(self.targets[0]) + ' = ' + render(self.value, brackets=False) + ';' if semicolon else ''

class Attribute(expr):
    """Attribute"""

    @classmethod
    def _from_py(cls, attr):
        return cls()

class AugAssign(stmt):
    """
    cboost AugAssign
    """
    target: list
    op: AST
    value: AST

    def __init__(self, target=None, op=None, value=None):
        self.target = target
        self.op = op
        self.value = value

    @classmethod
    def _from_py(cls, a: ast.Assign):
        """
        Convert Python AST AugAssign
        """
        return cls(
            target=convert(a.target),
            op=convert(a.op),
            value=convert(a.value)
        )

    def _render(self, curr_indent: str = '', semicolon = True, **kwargs):
        return curr_indent + render(self.target) + ' ' + render(self.op) + '= ' + render(self.value, brackets=False) + ';' if semicolon else ''

class BinOp(expr):
    """
    cboost BinOp
    """
    left: expr
    right: expr
    op: expr

    def __init__(self, left: expr = None, op: expr = None, right: expr = None):
        self.left = left
        self.op = op
        self.right = right

    @classmethod
    def _from_py(cls, bo: ast.BinOp):
        return cls(**{
            'left': convert(bo.left),
            'op': convert(bo.op),
            'right': convert(bo.right)
        })

    def _render(self, brackets: bool = True, **kwargs):
        return ''.join([
            '(' if brackets else '',
            render(self.left),
            ' ',
            render(self.op),
            ' ',
            render(self.right),
            ')' if brackets else ''
        ])

class BitAnd(operator):
    r: str = '&'

class BitOr(operator):
    r: str = '|'

class BitXor(operator):
    r: str = '^'

class BoolOp(expr):
    """
    cboost BoolOp
    """
    op: expr
    values: list

    def __init__(self, op: expr = None, values: list = []):
        self.op = op
        self.values = values

    @classmethod
    def _from_py(cls, bo: ast.BinOp):
        return cls(**{
            'op': convert(bo.op),
            'values': convert_list(bo.values)
        })

    def _render(self, brackets: bool = True, **kwargs):
        return ''.join([
            '(' if brackets else '',
            (' ' + render(self.op) + ' ').join([
                render(v) for v in self.values
            ]),
            ')' if brackets else ''
        ])

class Call(expr):
    """
    cboost Call
    """
    func: expr
    args: list
    keywords: list # WTF?

    def __init__(self, func: expr = None, args: list = [], keywords: list = []):
        self.func = func
        self.args = args
        self.keywords = keywords

    @classmethod
    def _from_py(cls, call: ast.Call):
        return cls(**{
            'func': convert(call.func),
            'args': convert_list(call.args),
            'keywords': convert_list(call.keywords)
        })

    def _render(self, **kwargs):
        return ''.join([
            render(self.func),
            '(',
            ', '.join([render(a, brackets=False) for a in self.args]),
            ')'
        ])

class Compare(expr):
    """
    cboost AST compare
    """
    # TODO: 
    # Python allow for multiple comparisons in single expression:
    # a < b == b2 > c
    # effectively this should be considered equal to:
    # (a < b) && (b == b2) && (b2 > c)
    # but Python makes each expression evaluated only once
    # so following expression:
    # a() < b() == b2() > c()
    # cannot be translated to:
    # (a() < b() && (b() == b2()) && (b2() > c())
    # cause this will make some functions called twice
    # moreover, once any condition fails, remaining part should not be evaluated at all
    #
    # verdict: we don't support more than one comparator

    left: expr
    ops: list
    comparators: list

    def __init__(self, left: expr = None, ops: list = [], comparators: list = []):
        self.left = left
        self.ops = ops
        self.comparators = comparators

    @classmethod
    def _from_py(cls, cmpr: ast.Compare):
        if len(cmpr.comparators) > 1:
            raise Exception("Expressions with multiple comparisons are not supported yet")

        # Special case: In()
        if type(cmpr.ops[0]) == ast.In:
            try:
                elts: list = cmpr.comparators[0].elts
                return BoolOp(
                    op=Or(),
                    values=[Compare(
                        left=convert(cmpr.left),
                        comparators=[convert(e)],
                        ops=[Eq()]
                    ) for e in elts]
                )
            except AttributeError:
                raise Exception("Unable to rewrite 'in' comparison")

        return cls(**{
            'left': convert(cmpr.left),
            'ops': convert_list(cmpr.ops),
            'comparators': convert_list(cmpr.comparators)
        })

    def _render(self, brackets: bool = True, **kwargs):
        return ''.join([
            '(' if brackets else '',
            render(self.left),
            ' ',
            render(self.ops[0]),
            ' ',
            render(self.comparators[0]),
            ')' if brackets else ''
        ])

class Constant(expr):
    """
    cboost AST constant
    """
    value: object
    def __init__(self, value: object=None):
        self.value = value

    @classmethod
    def _from_py(cls, const: ast.Constant):
        """
        convert AST Constant
        """
        # TODO:
        # some python constants needs conversion to C/C++ literals, types etc
        # - booleans !!!
        # - None -> nullptr !!
        # - big ints (https://faheel.github.io/BigInt/)
        # - ...

        value = const.value
        if value in [True, False]:
            value = int(value)
        if value == None:
            value = 0
        c = cls(value=value)
        return c

    def _render(self, **kwargs):
        match self.value:
            case int(x)|float(x):
                return str(x)
            case str(x):
                # TODO: string class!!!
                return '"'+x.replace('"', '\\"')+'"'

class Eq(cmpop):
    r: str = '=='

class For(stmt):
    """
    cboost AST For
    """
    target: expr
    iter: expr
    body: list
    def __init__(self, target: expr=None, iter: expr = None, body: list=[]):
        self.target = target
        self.body = body
        self.iter = iter

    @classmethod
    def _from_py(cls, fo: ast.While):
        if len(fo.orelse):
            raise Exception("For loops with strange 'else:' block are not supported yet")

        return cls(
            target=convert(fo.target),
            iter=convert(fo.iter),
            body=convert_list(fo.body)
        )

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        return '\n'.join([
            curr_indent + 'for (auto ' + render(self.target) + ': ' + render(self.iter) + '){',
            *[render(b, indent, next_indent) for b in self.body],
            curr_indent + '}'
        ])

class FunctionDef(stmt):
    name: str
    args: arguments
    body: list
    decorator_list: list
    returns: expr

    def __init__(self, name: str = '', args: arguments = None, body: list = [], decorator_list = None, returns = None):
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns

    @classmethod
    def _from_py(cls, fd: ast.FunctionDef):
        return cls(**{
            'name': fd.name,
            'args': convert(fd.args),
            'body': convert_list(fd.body),
            'decorator_list': convert_list(fd.decorator_list),
            'returns': convert(fd.returns)
        })

    def _render_declaration(self):
        try:
            return_type = convert_type(self.returns.id)
        except AttributeError:
            raise Exception('Functions without return type annotations are not supported')
        return return_type + ' ' + self.name + '(' + render(self.args) + ')'

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        declaration = self._render_declaration()
        return '\n'.join([
            curr_indent + declaration,
            curr_indent + '{',
            *[render(b, indent, next_indent) for b in self.body],
            curr_indent + '}',
            ''
        ])


class Gt(cmpop):
    r: str = '>'

class In(cmpop):
    """This shouldn't be used at all"""

class If(stmt):
    "cboost If"
    test: expr = None
    body: list = None
    orelse: list = None

    def __init__(self, test: expr = None, body: list = [], orelse: list = []):
        self.test = test
        self.body = body
        self.orelse = orelse

    @classmethod
    def _from_py(cls, the_if: ast.If):
        test = convert(the_if.test)
        body = convert_list(the_if.body)
        orelse = convert_list(the_if.orelse)
        i = cls(test=test, body=body, orelse=orelse)
        return i

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        return '\n'.join([
            curr_indent + 'if (' + render(self.test, brackets=False) + '){',
            *[render(b, indent, next_indent) for b in self.body],
            curr_indent + '}',
            *(
            [
                curr_indent + 'else {',
                *[render(b, indent, next_indent) for b in self.orelse],
                curr_indent + '}'
            ] if len(self.orelse) else [])
        ])

class Lt(cmpop):
    r = '<'

class LtE(cmpop):
    r = '<='

class Mod(operator):
    r: str = '%'

class Module(mod):
    """
    cboost AST Module
    """
    body: list
    def __init__(self, body: list=[]):
        self.body = body

    @classmethod
    def _from_py(cls, module: ast.Module):
        """
        Convert Python AST Module
        """
        body = convert_list(module.body)
        m = cls(body=body)
        return m

    def _render(self, indent: int = 4, curr_indent: str = '', **kwargs):
        return '\n'.join([
            f'// Module translated by cboost',
            'extern "C" {',
            *[f._render_declaration() +';' for f in self.body if isinstance(f, FunctionDef)],
            '}',
            *[render(b, indent, curr_indent) for b in self.body],
            f'// End of module'
        ])

class Mult(operator):
    """
    mult operator
    """
    r: str = '*'

class Name(expr):
    """
    cboost AST Name
    """
    id: str
    ctx: object
    def __init__(self, id: str='', ctx: object = None):
        self.id = id
        self.ctx = ctx

    @classmethod
    def _from_py(cls, name: ast.Name):
        """
        convert Python AST Name
        """
        id = name.id
        n = cls(id=id)
        return n

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        return self.id

class Or(boolop):
    r: str = '||'

class Return(stmt):
    value: ast.expr

    def __init__(self, value: ast.expr = None):
        self.value = value

    @classmethod
    def _from_py(cls, r: ast.Return):
        return cls(**{
            'value': convert(r.value)
        })

    def _render(self, curr_indent: str = '', **kwargs):
        return curr_indent + 'return ' + render(self.value, brackets=False) + ';'

class Sub(operator):
    """
    cboost Sub
    """
    r: str = '-'

class While(stmt):
    """
    cboost AST While
    """
    test: expr
    body: list
    def __init__(self, test: expr=None, body: list=[]):
        self.test = test
        self.body = body

    @classmethod
    def _from_py(cls, wh: ast.While):
        if len(wh.orelse):
            raise Exception("While loops with strange 'else:' block are not supported yet")
        return cls(**{
            'test': convert(wh.test),
            'body': convert_list(wh.body)
        })

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        return '\n'.join([
            curr_indent + 'while (' + render(self.test, brackets=False) + '){',
            *[render(b, indent, next_indent) for b in self.body],
            curr_indent + '}'
        ])

def __convert_class(py_ast_class: type) -> type:
    """
    Convert Python AST class to cboost AST class
    """
    global __class_conversions
    try:
        return __class_conversions[py_ast_class]
    except KeyError as e:
        raise Exception(f'Class not supported: {py_ast_class}')

def dump(obj: AST, indent: int = 0, __current_indent: int = 0) -> str:
    ii = ' ' * (indent + __current_indent)
    if isinstance(obj, AST):
        def is_simple(obj):
            return all(type(v) != list for v in vars(obj).values()) and (len(vars(obj)) <= 4)
        if is_simple(obj):
            indent, ii = 0, ''
        return (
            f'{type(obj).__name__}'
            + '('
            + ('\n' if indent else '')
            + (',\n' if indent else ', ').join(
                f'{ii}{p}=' + dump(v, indent=indent, __current_indent=__current_indent + indent) for (p, v) in vars(obj).items() if v != None
            )
            + ')'
        )
    elif isinstance(obj, list):
        if len(obj) == 0:
            indent, ii = 0, ''
        return (
            '['
            + ('\n' if indent else '')
            + (',\n' if indent else ', ').join(
                ii + dump(e, indent=indent, __current_indent=__current_indent + indent) for e in obj
                )
            + ']'
        )
    else:
        return repr(obj)

def render(obj: AST, indent: int = 4, curr_indent: str = '', brackets: bool = True) -> str:
    return obj._render(indent=indent, curr_indent=curr_indent, next_indent=curr_indent+indent*' ', brackets=brackets)

def convert(py_ast_object: ast.AST) -> AST:
    """
    Convert Python AST object to cboost AST
    """
    if py_ast_object == None:
        return None
    kls = __convert_class(type(py_ast_object))
    ast_obj = kls._from_py(py_ast_object)
    return ast_obj

def convert_list(element_list: list) -> list:
    """
    Convert list of Python AST objects to cboost AST objects
    """
    return [convert(e) for e in element_list]

def convert_type(type_name: str) -> str:
    """
    Convert type
    """
    try:
        return __type_conversions[type_name]
    except KeyError:
        return type_name + ' /* FIXME: Unknown type */'

__class_conversions: dict = {
    ast.arg:        arg,
    ast.arguments:  arguments,
    ast.Add:        Add,
    ast.And:        And,
    ast.AnnAssign:  AnnAssign,
    ast.Assign:     Assign,
    ast.Attribute:  Attribute,
    ast.AugAssign:  AugAssign,
    ast.BinOp:      BinOp,
    ast.BitAnd:     BitAnd,
    ast.BitOr:      BitOr,
    ast.BitXor:     BitXor,
    ast.BoolOp:     BoolOp,
    ast.Call:       Call,
    ast.Compare:    Compare,
    ast.Constant:   Constant,
    ast.Eq:         Eq,
    ast.For:        For,
    ast.FunctionDef:    FunctionDef,
    ast.Gt:         Gt,
    ast.If:         If,
    ast.In:         In,
    ast.Lt:         Lt,
    ast.LtE:        LtE,
    ast.Mod:        Mod,
    ast.Module:     Module,
    ast.Mult:       Mult,
    ast.Name:       Name,
    ast.Or:         Or,
    ast.Return:     Return,
    ast.Sub:        Sub,
    ast.While:      While,
}

__type_conversions: dict = {
    'bool':         'int',
    'int':          'long',
    'float':        'double',
    'str':          'string',
}

def make_cpp(func: object) -> object:
    global _disabled
    if _disabled:
        return func

    name = _make_cpp(func)

    def cboost_wrapper(*args, **kwargs):
        return call(name, args, kwargs)

    return cboost_wrapper

def find_function_name(py_src):
    """FIXME"""
    return ast.parse(py_src).body[0].name

def _make_cpp(func: object) -> str:
    global _boosted_py_src
    python_src: str = inspect.getsource(func)
    _boosted_py_src += python_src + '\n\n'
    name = find_function_name(python_src)
    _boosted_functions[name] = None
    return name

def call(name: str, args: list, kwargs: dict):
    f = _boosted_functions[name]
    if f == None:
        _load_so()
        f = _boosted_functions[name] # once again
    return f(*args, **kwargs)

def _cpp_id(cpp_src: str) -> str:
    return hashlib.sha256(cpp_src.encode()).hexdigest()[0:24]

def _load_so():
    global _boosted_py_src
    python_ast: ast.AST = ast.parse(_boosted_py_src)
    cpp_ast: AST = convert(python_ast)
    cpp_src: str = render(cpp_ast)

    cpp_id = _cpp_id(cpp_src)

    dirname = '__pycache__/__cboost__/'
    os.makedirs(dirname, exist_ok=True)

    fn_basename = f'{dirname}/{cpp_id}'
    fn_so = f'{fn_basename}.so'

    if not os.path.exists(fn_so):
        fn_cpp = f'{fn_basename}.cpp'
        with open(fn_cpp, 'w') as f:
            f.write(cpp_src)
        gcc_cmd = f'g++ -Wall -O3 -shared -o {fn_so} {fn_cpp}'
        gcc_ret = os.system(gcc_cmd)
        if gcc_ret != 0:
            raise Exception('C++ compilation failed.')

    so = ctypes.cdll.LoadLibrary(fn_so)

    for name in _boosted_functions.keys():
        _boosted_functions[name] = so[name]

