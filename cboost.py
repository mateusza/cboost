#!/usr/bin/env python3.10

import inspect
import ast
import dis
import os
import hashlib
import ctypes

ALL_FUNCTIONS = {}
IS_COMPILED = False
SO = None


class AST:
    """
    cboost Abstract Syntax Tree element
    """
    def __render__(self):
        """
        Render element into C/C++ source code
        """
        return f'/* AST {self} */'

    @classmethod
    def _from_py(cls, ast_element: ast.AST):
        """
        Create new element by converting Python AST Element
        """
        print(f'{ast_element=}')
        print(f'{type(ast_element)=}')
        print(f'{ast.dump(ast_element, indent=8)}')  
        raise Exception(f'Not implemented {cls=}')

class cmpop(AST):
    """
    cboost AST comparing operation
    """

    @classmethod
    def _from_py(cls, the_op: ast.cmpop):
        return cls()

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

class stmt(AST):
    """
    cboost AST statement
    """

class AnnAssign(stmt):
    """
    cboost AnnAssign
    """
    target: object = None
    annotation: object = None
    value: object = None
    simple: int = 0

    def __init__(self, target: AST = None, annotation: AST = None, value: expr=None, simple:int =0):
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
        targets = convert_list(assign.targets)
        value = convert(assign.value)
        a = cls(targets=targets, value=value)
        return a

class Compare(expr):
    """
    cboost AST compare
    """
    left: expr
    ops: list
    comparators: list

    def __init__(self, left: expr = None, ops: list = [], comparators: list = []):
        self.left = left
        self.ops = ops
        self.comparators = comparators

    @classmethod
    def _from_py(cls, cmpr: ast.Compare):
        return cls(**{
            'left': convert(cmpr.left),
            'ops': convert_list(cmpr.ops),
            'comparators': convert_list(cmpr.comparators)
        })

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

class Eq(cmpop):
    ...

class Gt(cmpop):
    ...

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

class Lt(cmpop):
    ...

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
    def is_simple(obj):
        return all(type(v) != list for v in vars(obj).values()) and len(vars(obj)) <= 3
    if isinstance(obj, AST):
        if is_simple(obj):
            indent = 0
            ii = ''
        return (
            f'{type(obj).__name__}'
            + '('
            + ('\n' if indent else '')
            + (',\n' if indent else ', ').join([f'{ii}{p}=' + dump(v, indent=indent, __current_indent=__current_indent + indent) for (p, v) in vars(obj).items()])
            + ')'
        )
    elif isinstance(obj, list):
        return (
            '['
            + ('\n' if indent else '')
            + (',\n' if indent else ', ').join(ii + dump(e, indent=indent, __current_indent=__current_indent + indent) for e in obj)
            + ']'
        )
    else:
        return repr(obj)

def convert(py_ast_object: ast.AST) -> AST:
    """
    Convert Python AST object to cboost AST
    """
    kls = __convert_class(type(py_ast_object))
    ast_obj = kls._from_py(py_ast_object)
    return ast_obj

def convert_list(element_list: list) -> list:
    """
    Convert list of Python AST objects to cboost AST objects
    """
    return [convert(e) for e in element_list]

__class_conversions: dict = {
    ast.AnnAssign:  AnnAssign,
    ast.Assign:     Assign,
    ast.Compare:    Compare,
    ast.Constant:   Constant,
    ast.Eq:         Eq,
    ast.Gt:         Gt,
    ast.If:         If,
    ast.Lt:         Lt,
    ast.Module:     Module,
    ast.Name:       Name,
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





