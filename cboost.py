#!/usr/bin/env python3.10

import inspect
import ast
import dis
import os
import hashlib
import ctypes
import dataclasses

ALL_FUNCTIONS = {}
IS_COMPILED = False
SO = None

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

    def _render(self, curr_indent: str = '', brackets: bool = True, **kwargs):
        return curr_indent + ''.join([
            '(' if brackets else '',
            render(self.left),
            ' ',
            render(self.op),
            ' ',
            render(self.right),
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
        return render(self.func) + '(' + ', '.join([render(a, brackets=False) for a in self.args]) + ')'

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
        return cls(**{
            'left': convert(cmpr.left),
            'ops': convert_list(cmpr.ops),
            'comparators': convert_list(cmpr.comparators)
        })

    def _render(self, curr_indent: str = '', brackets: bool = True, **kwargs):
        return curr_indent + ''.join([
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

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', brackets: bool = True):
        match self.value:
            case int(x)|float(x):
                return str(x)
            case str(x):
                # TODO: string class!!!
                return '"'+x.replace('"', '\\"')+'"'

class Eq(cmpop):
    r: str = '=='

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
            f'// Module: {self} translated by cboost',
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
    ast.AnnAssign:  AnnAssign,
    ast.Assign:     Assign,
    ast.BinOp:      BinOp,
    ast.Call:       Call,
    ast.Compare:    Compare,
    ast.Constant:   Constant,
    ast.Eq:         Eq,
    ast.FunctionDef:    FunctionDef,
    ast.Gt:         Gt,
    ast.If:         If,
    ast.Lt:         Lt,
    ast.LtE:        LtE,
    ast.Module:     Module,
    ast.Mult:       Mult,
    ast.Name:       Name,
    ast.Return:     Return,
    ast.Sub:        Sub,
    ast.While:      While,
}

__type_conversions: dict = {
    'int':          'long',
    'float':        'double',
    'str':          'string',
}

def translate_while_loop(wloop, indent: int=0) -> str:
    ind = " " * 4
    body = translate_body(wloop.body, indent=indent+4)
    cond = translate_instruction(wloop.test, semicolon=False)

    return ind + f'while({cond})' + '{\n' + body + '\n' + ind + '}\n'

def translate_range(r) -> (object, object, object):
    match r.iter.args:
        case [stop]:
            sss = [ast.Constant(value=0), stop, ast.Constant(value=1)]
        case [start, stop]:
            sss = [start, stop, ast.Constant(value=1)]
        case [start, stop, step]:
            sss = [start, stop, step]
    start, stop, step = sss
    init = ast.AnnAssign(
        target = r.target,
        value = start,
        annotation = ast.Name(id='int')
    )
    cond = ast.BinOp(
        left = r.target,
        right = stop,
        op = ast.Lt()
    )
    inc = ast.AugAssign(
        target = r.target,
        op = ast.Add(),
        value = step
    )
    return init, cond, inc

def translate_for_loop(floop, indent: int=0) -> str:
    ind = ' ' * indent
    body = translate_body(floop.body, indent=indent+4)
    try:
        assert type(floop.iter) == ast.Call
        assert floop.iter.func.id == 'range'
        init, cond, inc = translate_range(floop)
    except AssertionError:
        raise Error(f'Not supported. /* {ast.dump(floop)} */')

    init = translate_instruction(init)
    cond = translate_instruction(cond)
    inc = translate_instruction(inc, semicolon=False)

    return ind + f'for ({init} {cond} {inc})' + '{\n' + body + '\n' + ind + '}\n'

def translate_augassign(asgn) -> str:
    vname = asgn.target.id
    vval = translate_value(asgn.value)
    sign = translate_op(asgn.op)
    return f'{vname} {sign}= {vval}'

def translate_assign(asgn) -> str:
    match len(asgn.targets):
        case 1:
            vname = asgn.targets[0].id
            vval = translate_value(asgn.value)
            return f"{vname} = {vval}"
    return f'/* Multiple assign not supported {ast.dump(asgn)} */'

def translate_annassign(asgn) -> str:
    vname = asgn.target.id
    vtype = translate_type(asgn.annotation)
    if asgn.value == None:
        return f"{vtype} {vname}"
    vval = translate_value(asgn.value)
    return f"{vtype} {vname} = {vval}"

def translate_call(fcall) -> str:
    fname = fcall.func.id
    params = []
    for p in fcall.args:
        params.append(translate_value(p))
    return f"{fname}(" + ", ".join(params) + ")"

def translate_in_to_ors(cmpr: ast.Compare) -> str:
    values = [
       ast.Compare(left=cmpr.left, comparators=[elt], ops=[ast.Eq()])
       for elt in cmpr.comparators[0].elts
    ]
    
    new_or = ast.BoolOp(op=ast.Or(), values=values)
    return translate_boolop(new_or)

def translate_notin_to_and(cmpr: ast.Compare) -> str:
    values = [
       ast.Compare(left=cmpr.left, comparators=[elt], ops=[ast.NotEq()])
       for elt in cmpr.comparators[0].elts
    ]
    
    new_and = ast.BoolOp(op=ast.And(), values=values)
    return translate_boolop(new_and)

def translate_op(op) -> str:
    match op:
        case ast.Eq():
            return '=='
        case ast.NotEq():
            return '!='
        case ast.Lt():
            return "<"
        case ast.LtE():
            return "<="
        case ast.Gt():
            return '>'
        case ast.Add():
            return "+"
        case ast.Mult():
            return "*"
        case ast.Sub()|ast.USub():
            return "-"
        case ast.Mod():
            return '%'
        case ast.Or():
            return '||'
        case ast.And():
            return '&&'
        case ast.Div()|ast.FloorDiv():
            return '/'
    return f"/* UNKNOWN OPERATOR: {ast.dump(op)} */"

def translate_compare(cmpr: ast.Compare) -> str:
    match cmpr.ops[0]:
        case ast.In():
            return translate_in_to_ors(cmpr)
        case ast.NotIn():
            return translate_notin_to_and(cmpr)
    left = translate_value(cmpr.left)
    right = translate_value(cmpr.comparators[0])
    sign = translate_op(cmpr.ops[0])
    return f'({left} {sign} {right})'

def translate_constant(const) -> str:
    v = const.value
    match v:
        case bool(b):
            return '01'[int(b)]
        case int(i):
            return f'{i}'
        case float(i):
            return f'{i}'
    return f'CONSTANT /*{ast.dump(const)}*/';

def translate_binop(op) -> str:
    left = translate_value(op.left)
    right = translate_value(op.right)
    sign = translate_op(op.op)
    return f"({left} {sign} {right})"

def translate_unaryop(op) -> str:
    operand = translate_value(op.operand)
    sign = translate_op(op.op)
    return f"({sign} {operand})"

def translate_boolop(op) -> str:
    values2 = [translate_value(v) for v in op.values]
    sign = translate_op(op.op)
    return f' {sign} '.join(values2)

def translate_var(value) -> str:
    return value.id

def translate_value(value) -> str:
    match value:
        case ast.Name():
            return translate_var(value)
        case ast.BinOp():
            return translate_binop(value)
        case ast.UnaryOp():
            return translate_unaryop(value)
        case ast.BoolOp():
            return translate_boolop(value)
        case ast.Constant():
            return translate_constant(value)
        case ast.Call():
            return translate_call(value)
        case ast.Compare():
            return translate_compare(value)
    return f'VALUE: {ast.dump(value)}'

def translate_test(the_test) -> str:
    return translate_value(the_test)

def translate_if(instruction, indent: int=0) -> str:
    ind = " " * indent
    condition = translate_test(instruction.test)
    body = translate_body(instruction.body, indent=indent)
    return ind + "if(" + condition + "){\n" + body + "\n" + ind + "}"

def translate_return(instruction) -> str:
    value = translate_value(instruction.value)
    return f'return {value}'

def translate_type(annotation) -> str:
    match annotation.id:
        case 'int':
            return 'long'
        case 'float':
            return 'double'
        case 'bool':
            return 'int'
    return f'UNKNOWN_TYPE /*{ast.dump(annotation)}*/'

def translate_body(body, indent: int=0) -> str:
    instructions = []
    for instr in body:
        i = translate_instruction(instr, indent=indent)
        instructions.append(indent*" " + i)
    return "\n".join(instructions)

def translate_instruction(ins, semicolon=True, indent: int=0) -> str:
    sc = ";" if semicolon else ""
    ind = " " * indent
    match ins:
        case ast.If():
            return ind + translate_if(ins, indent=indent + 4)
        case ast.While():
            return ind + translate_while_loop(ins, indent=indent + 4)
        case ast.Return():
            return ind + translate_return(ins) + sc
        case ast.AnnAssign():
            return ind + translate_annassign(ins) + sc
        case ast.Assign():
            return ind + translate_assign(ins) + sc
        case ast.AugAssign():
            return ind + translate_augassign(ins) + sc
        case ast.For():
            return ind + translate_for_loop(ins, indent=indent + 4)
    return ind + translate_value(ins) + sc

def translate_arg(arg: ast.arg) -> str:
    name = arg.arg
    the_type = translate_type(arg.annotation)
    return f'{the_type} {name}'

def translate_args(arguments: ast.arguments) -> str:
    args = []
    for a in arguments.args:
        aa = translate_arg(a)
        args.append(aa)
    return ", ".join(args)

def get_arg_types(arguments: ast.arguments) -> list:
    args = []
    for a in arguments.args:
        aa = translate_arg(a)
        args.append(aa)
    return ", ".join(args)


def translate_function(fcode: ast.FunctionDef) -> (str, str, str):
    function_name: str = fcode.name
    return_type: str = translate_type(fcode.returns)
    arguments: str = translate_args(fcode.args)
    instructions: str = translate_body(fcode.body)
    hdr: str = f'{return_type} {function_name} ({arguments})'
    body: str = hdr + '{\n' + instructions + '\n}\n'
    return function_name, hdr+';', body

def translate_module(mod: ast.Module) -> (str, str, str):
    element = mod.body[0]
    match element:
        case ast.FunctionDef():
            return translate_function(element)
    return '???', '???', f'/* unknown element: {ast.dump(element)} */'

def process_python_func(func: object) -> str:
    python_src = inspect.getsource(func)
    tree = ast.parse(python_src)
    name, hdr, c = translate_module(tree)
    ALL_FUNCTIONS[name] = {"n": name, "h": hdr, "c": c, "ptr": None}
    return name

def make_c(func: object) -> object:
    name = process_python_func(func)

    def new_func(*args, **kwargs):
        return execute_c_func(name, args, kwargs)

    return new_func

def execute_c_func(name: str, args, kwargs):
    global LIBNAME
    if not IS_COMPILED:
        compile_c()
    f = ALL_FUNCTIONS[name]['ptr']
    return f(*args, **kwargs)
    
def make_c_code() -> str:
    hdrs = (['extern "C" {'] +
           [f['h'] for f in ALL_FUNCTIONS.values()] +
           ['}'])
    code = [f['c'] for f in ALL_FUNCTIONS.values()]
    return "\n".join(hdrs + code)

def compile_c():
    c_code = make_c_code()
    global IS_COMPILED
    fid = hashlib.sha256(c_code.encode()).hexdigest()[0:16]
    prefix = '__pycache__/cboost-'
    tc = f'{prefix}{fid}.cpp'
    tso = f'{prefix}{fid}.so'
    tlog = f'{prefix}{fid}.log'
    with open(tc, 'w') as f:
        f.write(c_code)
    cmd = f'g++ -Wall -O3 -shared -o {tso} {tc} > {tlog}'
    ret = os.system(cmd)
    assert ret == 0
    SO = ctypes.cdll.LoadLibrary(tso)
    IS_COMPILED = True
    for n in ALL_FUNCTIONS.keys():
        ALL_FUNCTIONS[n]['ptr'] = SO[n]





