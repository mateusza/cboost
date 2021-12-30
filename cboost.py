#!/usr/bin/env python3.10

import inspect
import ast
import dis
import os
import tempfile
import ctypes

ALL_FUNCTIONS = {}
IS_COMPILED = False
LIBNAME = None
SO = None

def translate_while_loop(wloop) -> str:
    body = translate_body(wloop.body)
    cond = translate_instruction(wloop.test, semicolon=False)

    return f'while({cond})' + '{\n' + body + '\n}\n'

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

def translate_for_loop(floop) -> str:
    body = translate_body(floop.body)
    try:
        assert type(floop.iter) == ast.Call
        assert floop.iter.func.id == 'range'
        init, cond, inc = translate_range(floop)
    except AssertionError:
        raise Error(f'Not supported. /* {ast.dump(floop)} */')

    init = translate_instruction(init)
    cond = translate_instruction(cond)
    inc = translate_instruction(inc, semicolon=False)

    return f'for ({init} {cond} {inc})' + '{\n' + body + '\n}'

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
        case ast.Gt():
            return '>'
        case ast.Add():
            return "+"
        case ast.Mult():
            return "*"
        case ast.Sub():
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

def translate_if(instruction) -> str:
    condition = translate_test(instruction.test)
    body = translate_body(instruction.body)
    return "if(" + condition + "){\n" + body + "\n}"

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

def translate_body(body) -> str:
    instructions = []
    for instr in body:
        i = translate_instruction(instr)
        instructions.append(i)
    return "\n".join(instructions)

def translate_instruction(ins, semicolon=True) -> str:
    sc = ";" if semicolon else ""
    match ins:
        case ast.If():
            return translate_if(ins)
        case ast.While():
            return translate_while_loop(ins)
        case ast.Return():
            return translate_return(ins) + sc
        case ast.AnnAssign():
            return translate_annassign(ins) + sc
        case ast.Assign():
            return translate_assign(ins) + sc
        case ast.AugAssign():
            return translate_augassign(ins) + sc
        case ast.For():
            return translate_for_loop(ins)
    return translate_value(ins) + sc

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
    argument_typs = get_arg_types(fcode.args)
    instructions: str = translate_body(fcode.body)
    hdr: str = f'{return_type} {function_name} ({arguments})'
    body: str = hdr + '{\n' + instructions + '\n}\n'
    return function_name, hdr+';', body, [return_type, argument_types(fcode

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
        c_code = make_c_code()
        LIBNAME = compile_c(c_code)
    f = ALL_FUNCTIONS[name]['ptr']
    return f(*args, **kwargs)
    
def make_c_code() -> str:
    hdrs = [f['h'] for f in ALL_FUNCTIONS.values()]
    code = [f['c'] for f in ALL_FUNCTIONS.values()]
    return "\n".join(hdrs + code)

def compile_c(c_code):
    global IS_COMPILED
    tc = tempfile.mktemp('.c')
    tso = tempfile.mktemp('.so')
    with open(tc, 'w') as f:
        f.write(c_code)
    print(f"-----BEGIN C-----\n{c_code}\n-----END C-----")
    cmd = f'tcc -shared -o {tso} {tc}'
    print(f"sys: {cmd}")
    ret = os.system(cmd)
    assert ret == 0
    print(tso)
    SO = ctypes.cdll.LoadLibrary(tso)
    IS_COMPILED = True
    for n in ALL_FUNCTIONS.keys():
        ALL_FUNCTIONS[n]['ptr'] = SO[n]

