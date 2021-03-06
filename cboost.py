#!/usr/bin/env python3

import inspect
import ast
import dis
import os
import hashlib
import ctypes
import sys

_boosted_functions: dict = {}
_boosted_py_src: str = ''
_disabled = False
_cache = True
_compiler = 'g++'
_compiler_opts = ['-fPIC', '-Werror', '-O2']

if os.sys.platform == 'darwin':
    _compiler_opts += ['-std=c++11']

_cpp_functions = open('cboost.hpp').read()

def disable():
    global _disabled
    _disabled = True

def _is_disabled() -> bool:
    global _disabled
    if _disabled:
        return True
    if os.getenv('CBOOST_DISABLE'):
        sys.stderr.write('Warning: cboost disabled by CBOOST_DISABLE\n')
        return True
    return False

def _use_cache() -> bool:
    global _cache
    if _cache == False:
        return False
    if os.getenv('CBOOST_NOCACHE'):
        sys.stderr.write('Warning: cboost cache disabled by CBOOST_NOCACHE\n')
        return False
    return True

def add_brackets(render_method):
    def fn(what, brackets: bool=True, **kwargs):
        return ('(' if brackets else '') + render_method(what, **kwargs) +  (')' if brackets else '')
    return fn

def add_semicolon(render_method):
    def fn(what, semicolon: bool=True, **kwargs):
        return render_method(what, **kwargs) +  (';' if semicolon else '')
    return fn

def add_indent(render_method):
    def fn(what, curr_indent: str='', **kwargs):
        return curr_indent + render_method(what, **kwargs)
    return fn

def escape_string_literal(s: str):
    return s.replace('"', '\\"')

class AST:
    """
    cboost Abstract Syntax Tree element
    """

    @add_indent
    def _render(self, **kwargs) -> str:
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
        arg = a.arg
        annotation = convert(a.annotation)
        return cls(arg=arg, annotation=annotation)

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

    def __init__(self, *, args, posonlyargs, kwonlyargs, kw_defaults, defaults):
        self.args = args
        self.posonlyargs = posonlyargs
        self.kwonlyargs = kwonlyargs
        self.kw_defaults = kw_defaults
        self.defaults = defaults

    @classmethod
    def _from_py(cls, a: ast.arguments):
        args = convert_list(a.args)
        posonlyargs = convert_list(a.posonlyargs)
        kwonlyargs = convert_list(a.kwonlyargs)
        kw_defaults = convert_list(a.kw_defaults)
        defaults = convert_list(a.defaults)

        return cls(args=args, posonlyargs=posonlyargs, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, defaults=defaults)

    def _render(self, **kwargs):
        return render_list(self.args)

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
    def _from_py(cls, a: ast.AnnAssign):
        """
        convert
        """
        target = convert(a.target)
        simple = a.simple
        value = convert(a.value)
        annotation_id = str(a.annotation.value) if type(a.annotation) == ast.Constant else convert_type(a.annotation.id)
        annotation = Name(id=annotation_id)
        return cls(target=target, annotation=annotation, value=value, simple=simple)

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return render(self.annotation) + ' ' + render(self.target) + ' = ' + render(self.value, brackets=False)


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
        return cls(targets=targets, value=value)

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return render(self.targets[0], brackets=False) + ' = ' + render(self.value, brackets=False)

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
        target = convert(a.target)
        op = convert(a.op)
        value = convert(a.value)
        return cls(target=target, op=op, value=value)

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return render(self.target) + ' ' + render(self.op) + '= ' + render(self.value, brackets=False)

class BinOp(expr):
    op: expr
    values: list

    def __init__(self, op: expr = None, values: list = []):
        self.op = op
        self.values = values

    @add_brackets
    def _render(self, **kwargs):
        return render_list(self.values, separator=' ' + render(self.op) + ' ', brackets=False)

    @classmethod
    def _from_py(cls, bo: ast.BinOp):
        values = [convert(bo.left), convert(bo.right)]
        op = convert(bo.op)
        return cls(op=op, values=values)

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
    def _from_py(cls, bo: ast.BoolOp):
        op = convert(bo.op)
        values = convert_list(bo.values)
        return cls(op=op, values=values)

    @add_brackets
    def _render(self, **kwargs):
        return render_list(self.values, separator=' ' + render(self.op) + ' ', brackets=False)

class Break(stmt):
    def __init__(self):
        ...

    @classmethod
    def _from_py(cls, b=None):
        return cls()

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return 'break'

class Call(expr):
    """
    cboost Call
    """
    func: expr
    args: list
    keywords: list # named args :-)

    def __init__(self, func: expr = None, args: list = [], keywords: list = []):
        self.func = func
        self.args = args
        self.keywords = keywords

    @classmethod
    def _from_py(cls, call: ast.Call):
        if len(call.keywords) != 0:
            raise Exception("Named arguments are not supported in C++")

        if type(call.func) == ast.Name: # qwe()
            func = convert(call.func)
            args = convert_list(call.args)
        elif type(call.func) == ast.Attribute: # something.join()
            funcname = 'py::methods::' + call.func.attr
            func = Name(id=funcname)
            args = [convert(call.func.value), *convert_list(call.args)]
        else:
            raise Exception("Unknown call: " + ast.dump(call))
        return cls(func=func, args=args)

    def _render(self, **kwargs):
        return render(self.func) + render_list(self.args)

class Compare(expr):
    """
    cboost AST compare
    """
    # TODO: 
    # Python allows for multiple comparisons in a single expression:
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

        left = convert(cmpr.left)
        ops = convert_list(cmpr.ops)
        comparators = convert_list(cmpr.comparators)

        return cls(left=left, ops=ops, comparators=comparators)

    @add_brackets
    def _render(self, **kwargs):
        return render(self.left) + ' ' + render(self.ops[0]) + ' ' + render(self.comparators[0])

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
        return cls(value=value)

    def _render(self, **kwargs):
        value_type = type(self.value)
        if value_type in (int, float):
            return str(self.value)
        if value_type in (str, ):
            return 'std::string{"'+escape_string_literal(self.value)+'"}'

class Continue(stmt):
    def __init__(self):
        ...

    @classmethod
    def _from_py(cls, c=None):
        return cls()

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return 'continue'

class Div(operator):
    """
    cboost Div
    """
    r: str = '/'

class Eq(cmpop):
    r: str = '=='

class Expr(stmt):
    """
    cboost Expr
    """
    value = None
    def __init__(self, value=None):
        self.value = value

    @classmethod
    def _from_py(cls, e: ast.Expr):
        """
        Convert Python AST Expr
        """
        value = convert(e.value)
        return cls(value=value)

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return render(self.value, brackets=False)

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
    def _from_py(cls, fo: ast.For):
        if len(fo.orelse):
            raise Exception("For loops with strange 'else:' block are not supported yet")

        if 1: # Convert for x in range(): to C-style for(;;)
            if type(fo.iter) == ast.Call and type(fo.iter.func) == ast.Name and fo.iter.func.id == 'range':
                return ForCStyle._from_py(fo)

        target = convert(fo.target)
        iter = convert(fo.iter)
        body = convert_list(fo.body)

        return cls(target=target, iter=iter, body=body)

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        hdr = 'for (auto ' + render(self.target) + ' : ' + render(self.iter) + ')'
        return render_block(hdr=hdr, body=self.body, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs)

class ForCStyle(stmt):
    init: stmt
    cond: expr
    incr: stmt
    body: list

    def __init__(self, init=None, cond=None, incr=None, body=[]):
        self.init = init
        self.cond = cond
        self.incr = incr
        self.body = body

    @classmethod
    def _from_py(cls, fo: ast.For):
        target = convert(fo.target)
        r_args = convert_list(fo.iter.args)

        if len(r_args) == 1:
            r_args = [Constant(0), *r_args]
        if len(r_args) == 2:
            r_args = [*r_args, Constant(1)]

        start, stop, step = r_args

        if type(step) == Constant and step.value == 1:
            incr = IncrDecr(target=target, op=Add(), post=False)
        else:
            incr = AugAssign(target=target, op=Add(), value=step)

        init = AnnAssign(target=target, annotation=Name(id='auto'), value=start, simple=1)
        cond = Compare(left=target, ops=[Lt()], comparators=[stop])
        body = convert_list(fo.body)
        return cls(init=init, cond=cond, incr=incr, body=body)

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        hdr = 'for (' + render(self.init, semicolon=False) + '; ' + render(self.cond, brackets=False) + '; ' + render(self.incr, semicolon=False, brackets=False) +')'
        return render_block(hdr=hdr, body=self.body, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs)

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
        
        name = fd.name
        args = convert(fd.args)
        body = convert_list(fd.body)
        decorator_list = convert_list(fd.decorator_list)
        returns = convert(fd.returns)
    
        return cls(name=name, args=args, body=body, decorator_list=decorator_list, returns=returns)

    def _render_declaration(self):
        try:
            return_type = convert_type(self.returns.id)
        except AttributeError:
            return_type = 'void'
        return return_type + ' ' + self.name + render(self.args)

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        declaration = self._render_declaration()
        return render_function(hdr=declaration, body=self.body, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs)

class Gt(cmpop):
    r: str = '>'

class IncrDecr(expr):
    target: expr
    op: operator
    post: bool

    def __init__(self, target: expr = None, op: operator = None, post: bool = True):
        self.target = target
        self.op = op
        self.post = post

    @add_brackets
    def _render(self, **kwargs):
        target = render(self.target)
        op = render(self.op)*2
        v = [target, op]
        if not self.post:
            v.reverse()
        a, b = v
        return a + b

class In(cmpop):
    """This shouldn't be used at all as C++ doesn't have direct syntactical equivalent"""
    def __init__(self):
        raise Exception('Use of In() is impossible')

class Index(AST):
    "Index"

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
        return cls(test=test, body=body, orelse=orelse)

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        hdr = 'if (' + render(self.test, brackets=False) + ')'
        elsehdr = 'else '
        ifblock = render_block(hdr=hdr, body=self.body, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs)
        elseblock = render_block(hdr=elsehdr, body=self.orelse, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs) if len(self.orelse) else ''
        return ifblock + elseblock

class IfExp(expr):
    test: expr
    body: expr
    orelse: expr

    def __init__(self, test: expr = None, body: expr = None, orelse: expr = None):
        self.test = test
        self.body = body
        self.orelse = orelse

    @classmethod
    def _from_py(cls, ife: ast.IfExp):
        test = convert(ife.test)
        body = convert(ife.body)
        orelse = convert(ife.orelse)
        return cls(test=test, body=body, orelse=orelse)

    @add_brackets
    def _render(self, **kwargs):
        return render(self.test) + ' ? ' + render(self.body) + ' : ' + render(self.orelse)

class JoinedStr(expr):
    values: list

    def __init__(self, values: list = []):
        self.values = values
    
    @add_brackets
    def _render(self, **kwargs):
        return render_list(self.values, separator=' + ', brackets=False)

    @classmethod
    def _from_py(cls, js: ast.JoinedStr()):
        values = []
        for v in js.values:
            if type(v) == ast.FormattedValue:
                v = v.value
            v = convert(v)
            if type(v) == Constant and type(v.value) == str:
                nv = v
            else:
                nv = Call(func=Name(id='str'), args=[v])
            values.append(nv)
        return cls(values=values)

class List(expr):
    elts: list

    def _render(self):
        raise Exception("Lists cannot be rendered directly.")

    @classmethod
    def _from_py(cls, the_list: ast.List):
        if any(type(e) == ast.Starred for e in the_list.elts):
            sublists = []
            queue = [*the_list.elts]
            while True:
                try:
                    spos = next(k for k, v in enumerate(queue) if type(v) == ast.Starred)
                except StopIteration:
                    sublists.append(Std_Vector(elts=convert_list(queue)))
                    break
                prefix = queue[0:spos]
                if len(prefix):
                    sublists.append(Std_Vector(elts=convert_list(prefix)))
                sublists.append(convert(queue[spos]))
                queue = queue[spos+1:]
                if len(queue) == 0:
                    break
            s = BinOp(op=Add(), values=sublists)
            return s

        if len(the_list.elts) == 0:
            raise Exception("Empty list literals are not supported as we cannot deduce the type")
        return Std_Vector._from_py_list(the_list)

class Lt(cmpop):
    r: str = '<'

class LtE(cmpop):
    r: str = '<='

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
        return cls(body=body)

    def _render(self, indent: int = 4, curr_indent: str = '', **kwargs):
        return '\n'.join([
            f'// Module translated by cboost',
            _cpp_functions,
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
        return cls(id=id)

    def _render(self, **kwargs):
        return self.id

class NotEq(cmpop):
    r: str = '!='

class Or(boolop):
    r: str = '||'

class Return(stmt):
    value: ast.expr

    def __init__(self, value: ast.expr = None):
        self.value = value

    @classmethod
    def _from_py(cls, r: ast.Return):
        value = convert(r.value)
        return cls(value=value)

    @add_semicolon
    @add_indent
    def _render(self, **kwargs):
        return 'return ' + render(self.value, brackets=False)

class Std_Vector(expr):
    elts: list
    cpp_type = 'std::vector'

    def __init__(self, elts: list):
        self.elts = elts

    @classmethod
    def _from_py_list(cls, the_list):
        elts = convert_list(the_list.elts)
        return cls(elts=elts)

    def _render(self, **kwargs):
        first_elt = self.elts[0]
        return self.cpp_type + '<decltype(' + render(first_elt) + ')>' + '{' + render_list(self.elts, brackets=False) + '}'

class Starred(expr):
    """Starred"""
    value: expr

    def __init__(self, value: expr=None):
        self.value = value

    @classmethod
    def _from_py(cls, st):
        value = convert(st.value)
        return cls(value=value)

    def _render(self, **kwargs):
        return 'py::operators::starred(' + render(self.value, brackets=False) + ')'

class Sub(operator):
    """cboost sub"""
    r: str = '-'

class Subscript(expr):
    value: expr
    slice: expr

    def __init__(self, value: expr, slice: expr):
        self.value = value
        self.slice = slice

    @classmethod
    def _from_py(cls, ss):
        value = convert(ss.value)
        if type(ss.slice) == ast.Index:
            slice = convert(ss.slice.value)
        else:
            slice = convert(ss.slice)
        return cls(value=value, slice=slice)

    @add_brackets
    def _render(self, **kwargs):
        return render(self.value) + '[' + render(self.slice, brackets=False) + ']'

class UnaryOp(expr):
    """
    cboost BinOp
    """
    operand: expr
    op: expr

    def __init__(self, operand: expr = None, op: expr = None):
        self.op = op
        self.operand = operand

    @classmethod
    def _from_py(cls, uo: ast.UnaryOp):
        op = convert(uo.op)
        operand = convert(uo.operand)
        return cls(op=op, operand=operand)

    @add_brackets
    def _render(self, **kwargs):
        return render(self.op) + ' ' + render(self.operand)

class USub(operator):
    """
    cboost sub
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
        test = convert(wh.test)
        body = convert_list(wh.body)
        return cls(test=test, body=body)

    def _render(self, indent: int = 4, curr_indent: str = '', next_indent: str = '', **kwargs):
        hdr = 'while(' + render(self.test, brackets=False) + ')'
        return render_block(hdr=hdr, body=self.body, hdr_indent=curr_indent, indent=indent, curr_indent=next_indent, **kwargs)

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

def render(obj: AST, indent: int=4, curr_indent: str = '', brackets: bool = True, semicolon: bool = True) -> str:
    return obj._render(indent=indent, curr_indent=curr_indent, next_indent=curr_indent+indent*' ', brackets=brackets, semicolon=semicolon)

def render_body(body: list, **kwargs):
    return "\n".join(render(b, **kwargs) for b in body)

def render_block(hdr: str, hdr_indent: str, curr_indent: str, body: list, **kwargs):
    brace1, brace2 = (' {', hdr_indent + '}\n') if len(body) != 1 else ('', '')
    return hdr_indent + hdr + brace1 + '\n' + render_body(body, curr_indent=curr_indent, **kwargs) + '\n' + brace2

def render_function(hdr: str, hdr_indent: str, curr_indent: str, body: list, **kwargs):
    return hdr_indent + hdr + ' {\n' + render_body(body, curr_indent=curr_indent, **kwargs) + '\n' + hdr_indent + '}\n'


@add_brackets
def render_list(elts: list, separator: str=', ', **kwargs):
    return separator.join(render(e, brackets=False) for e in elts)

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
    ast.Break:      Break,
    ast.Call:       Call,
    ast.Compare:    Compare,
    ast.Constant:   Constant,
    ast.Continue:   Continue,
    ast.Div:        Div,
    ast.Eq:         Eq,
    ast.Expr:       Expr,
    ast.For:        For,
    ast.FunctionDef:    FunctionDef,
    ast.Gt:         Gt,
    ast.If:         If,
    ast.IfExp:      IfExp,
    ast.In:         In,
    ast.Index:      Index,
    ast.JoinedStr:  JoinedStr,
    ast.List:       List,
    ast.Lt:         Lt,
    ast.LtE:        LtE,
    ast.Mod:        Mod,
    ast.Module:     Module,
    ast.Mult:       Mult,
    ast.NotEq:      NotEq,
    ast.Name:       Name,
    ast.Or:         Or,
    ast.Return:     Return,
    ast.Starred:    Starred,
    ast.Sub:        Sub,
    ast.Subscript:  Subscript,
    ast.UnaryOp:    UnaryOp,
    ast.USub:       USub,
    ast.While:      While,
}

__type_conversions: dict = {
    'bool':         'int32_t',
    'int':          'int64_t',
    'float':        'double',
    'str':          'std::string',
    'list':         'auto', # FIXME
}

def make_cpp(func: object) -> object:
    """
        The Holy Wrapper
    """
    if _is_disabled():
        return func

    name = _make_cpp(func)

    def cboost_wrapper(*args, **kwargs):
        return call(name, args, kwargs)

    return cboost_wrapper

def find_function_name(py_src):
    """FIXME"""
    return ast.parse(py_src).body[0].name

def _make_cpp(func: object) -> str:
    """
    Convert function code to C++, compile it to shared library and replace with library call.
    """
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

def _src_id(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[0:24]

def _load_so():
    global _boosted_py_src
    global _cache

    dirname = '__pycache__/__cboost__'

    py_id = _src_id(_boosted_py_src)
    fn_py_hash = f'{dirname}/{py_id}.txt'

    cpp_id: str = None

    if _use_cache():
        try:
            with open(fn_py_hash) as f:
                cpp_id = f.read().strip()
        except FileNotFoundError:
            ...

    if cpp_id == None:
        python_ast: ast.AST = ast.parse(_boosted_py_src)
        cpp_ast: AST = convert(python_ast)
        cpp_src = render(cpp_ast)
        cpp_id = _src_id(cpp_src)

    os.makedirs(dirname, exist_ok=True)

    fn_basename = f'{dirname}/{cpp_id}'
    fn_so = f'{fn_basename}.so'

    try:
        if _use_cache() == False:
            raise OSError
        so = ctypes.cdll.LoadLibrary(fn_so)
    except OSError as e:
        fn_cpp = f'{fn_basename}.cpp'
        fn_errors = f'{fn_basename}.log'

        with open(fn_cpp, 'w') as f:
            f.write(cpp_src)

        with open(fn_py_hash, 'w') as f:
            f.write(cpp_id)

        # TODO: compiler, options, flags, etc
        global _compiler
        gcc_cmd = f'{_compiler} {" ".join(_compiler_opts)} -shared -o {fn_so} {fn_cpp} 2> {fn_errors}'
        gcc_ret = os.system(gcc_cmd)
        if gcc_ret != 0:
            os.unlink(fn_py_hash)
            raise Exception(f'C++ compilation failed. See {fn_errors} for details and {fn_cpp} for generated C++ code.')
        os.unlink(fn_errors)
        so = ctypes.cdll.LoadLibrary(fn_so)

    for name in _boosted_functions.keys():
        _boosted_functions[name] = so[name]

