#!/usr/bin/env python3
"""Stub generator for C modules.

The public interface is via the mypy.stubgen module.
"""

import importlib
import inspect
import os.path
import re
from typing import List, Dict, Tuple, Optional, Mapping, Any, Set, overload
from types import ModuleType
from typing_extensions import Final

from mypy.moduleinspect import is_c_module
from mypy.stubdoc import (
    infer_sig_from_docstring, infer_prop_type_from_docstring, ArgSig,
    infer_arg_sig_from_anon_docstring, infer_ret_type_sig_from_anon_docstring,
    infer_ret_type_sig_from_docstring, FunctionSig
)
from mypy.stubgenc import *

import funcparser


def generate_stub_for_c_module_costum(module_name: str,
                                      target: str,
                                      sigs: Optional[Dict[str, str]] = None,
                                      class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for C module.
       (Custom modified version, prototype is mypy.stubgenc.generate_stub_for_c_module)

    This combines simple runtime introspection (looking for docstrings and attributes
    with simple builtin types) and signatures inferred from .rst documentation (if given).

    If directory for target doesn't exist it will be created. Existing stub
    will be overwritten.
    """
    module = importlib.import_module(module_name)
    assert is_c_module(module), '%s is not a C module' % module_name
    subdir = os.path.dirname(target)
    if subdir and not os.path.isdir(subdir):
        os.makedirs(subdir)
    imports: List[str] = []
    functions: List[str] = []
    done = set()
    items = sorted(module.__dict__.items(), key=lambda x: x[0])
    for name, obj in items:
        if is_c_function(obj):
            generate_c_function_stub_costum(
                module, name, obj, functions, imports=imports, sigs=sigs)
            done.add(name)
    types: List[str] = []
    for name, obj in items:
        if name.startswith('__') and name.endswith('__'):
            continue
        if is_c_type(obj):
            generate_c_type_stub_custom(module, name, obj, types, imports=imports, sigs=sigs,
                                        class_sigs=class_sigs)
            done.add(name)
    variables = []
    for name, obj in items:
        if name.startswith('__') and name.endswith('__'):
            continue
        if name not in done and not inspect.ismodule(obj):
            type_str = strip_or_import(
                get_type_fullname(type(obj)), module, imports)
            variables.append('%s: %s' % (name, type_str))
    output = []
    for line in sorted(set(imports)):
        output.append(line)
    for line in variables:
        output.append(line)
    for line in types:
        if line.startswith('class') and output and output[-1]:
            output.append('')
        output.append(line)
    if output and functions:
        output.append('')
    for line in functions:
        output.append(line)
    output = add_typing_import(output)
    with open(target, 'w') as file:
        for line in output:
            file.write('%s\n' % line)


def refine_func_signature(docstr: str, func_name: str, is_overload: bool = False, sigid: int = 0) -> str:
    """
    Trans docstring from c module to standard function signature 

    Args:
        docstr (str): docstring from c module
        func_name (str): name of function
        is_overload (bool, optional): function is overload . Defaults to False.
        sigid (int, optional): if function is overload it's signature id in docstring. Defaults to 0.

    Returns:
        str: standard function signature like: def func_name(args...) -> ret_type : ...
    """
    funcsig_reg = re.compile(
        (str(sigid) + ". " if is_overload else "") + func_name + r"\(.*?\) ->.*")
    match_str = re.match(funcsig_reg, docstr)
    if match_str:
        func_str = re.search(func_name+r"\(.*?\) ->.*", match_str.group(0))
        # complement signature to function declaration then give to python parser
        func_str = "def " + func_str.group(0) + ": ..."
        return func_str
    else:
        return None


def generate_c_function_stub_costum(module: ModuleType,
                                    name: str,
                                    obj: object,
                                    output: List[str],
                                    imports: List[str],
                                    self_var: Optional[str] = None,
                                    sigs: Optional[Dict[str, str]] = None,
                                    class_name: Optional[str] = None,
                                    class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for a single function or method.
       (Custom modified version, prototype is mypy.stubgenc.generate_c_function_stub)

    The result (always a single line) will be appended to 'output'.
    If necessary, any required names will be added to 'imports'.
    The 'class_name' is used to find signature of __init__ or __new__ in
    'class_sigs'.
    """
    # insert Set type from type for mypy missed it
    imports.append("from typing import Set")

    if sigs is None:
        sigs = {}
    if class_sigs is None:
        class_sigs = {}

    ret_type = 'None' if name == '__init__' and class_name else 'Any'

    if (
        name in ("__new__", "__init__")
        and name not in sigs
        and class_name
        and class_name in class_sigs
    ):
        inferred: Optional[List[FunctionSig]] = [
            FunctionSig(
                name=name,
                args=infer_arg_sig_from_anon_docstring(class_sigs[class_name]),
                ret_type=ret_type,
            )
        ]
    else:
        docstr = getattr(obj, '__doc__', None)
        inferred = infer_sig_from_docstring(docstr, name)
        if inferred:
            assert docstr is not None
            if is_pybind11_overloaded_function_docstring(docstr, name):
                # Remove pybind11 umbrella (*args, **kwargs) for overloaded functions
                del inferred[-1]
        if not inferred:
            if class_name and name not in sigs:
                inferred = [FunctionSig(name, args=infer_method_sig(name, self_var),
                                        ret_type=ret_type)]
            else:
                inferred = [FunctionSig(name=name,
                                        args=infer_arg_sig_from_anon_docstring(
                                            sigs.get(name, '(*args, **kwargs)')),
                                        ret_type=ret_type)]
        elif class_name and self_var:
            args = inferred[0].args
            if not args or args[0].name != self_var:
                args.insert(0, ArgSig(name=self_var))

    is_overloaded = len(inferred) > 1 if inferred else False
    if is_overloaded:
        imports.append('from typing import overload')
    #TODO: logic branch too deep, need split
    if inferred:
        # signature id for overload func, used to pick corresbonding signature from inferred docstring
        sigid = 0
        for signature in inferred:
            arg_sig = []
            # in docstring, overload function signature start from 1.
            sigid += 1
            for arg in signature.args:
                if arg.name == self_var:
                    arg_def = self_var
                else:
                    arg_def = arg.name
                    if arg_def == 'None':
                        arg_def = '_none'  # None is not a valid argument name

                    if arg.type:
                        arg_def += ": " + \
                            strip_or_import(arg.type, module, imports)

                    # get function default value from func signature in __doc__
                    if arg.default:
                        if is_overloaded:
                            doc = docstr.split("\n")[3: -1]
                            for i in range(0, len(doc)):
                                # get signature from overload function docstr
                                func_str = refine_func_signature(
                                    doc[i], name, is_overloaded, sigid)
                                if func_str:
                                    var_str = funcparser.getFuncVarStr(
                                        func_str, arg.name)
                                    default_var = re.search(
                                        r" = .{0,}", var_str)
                                    if default_var:
                                        # parsered default var may contains traill char ",", strip it
                                        arg_def += default_var.group(
                                            0).strip(",")
                                    else:
                                        arg_def += " = ..."
                                    break
                        else:
                            # similar like overload function
                            func_str = refine_func_signature(
                                docstr.split('\n')[0], name)
                            var_str = funcparser.getFuncVarStr(
                                func_str, arg.name)
                            default_var = re.search(r" = .{0,}", var_str)
                            if default_var:
                                arg_def += default_var.group(0).strip(",")
                            else:
                                arg_def += " = ..."

                arg_sig.append(arg_def)

            if is_overloaded:
                output.append('@overload')
            output.append('def {function}({args}) -> {ret}:'.format(
                function=name,
                args=", ".join(arg_sig),
                ret=strip_or_import(signature.ret_type, module, imports)
            ))
            # append function summary from __doc__
            output.append("    \"\"\"")
            if is_overloaded:
                doc = docstr.split("\n")[3: -1]
                for i in range(0, len(doc)):
                    funcsig_reg = re.compile(
                        str(sigid) + ". " + name + r"\(.*?\) ->.*")
                    next_funcsig_reg = re.compile(
                        str(sigid+1) + ". " + name + r"\(.*?\) ->.*")
                    if re.match(funcsig_reg, doc[i]):
                        for j in range(i+2, len(doc)):
                            if re.match(next_funcsig_reg, doc[j]):
                                break
                            output.append(
                                '    {docline}'.format(docline=doc[j]))
                        break
            else:
                funcsig_reg = re.compile(name + r"\(.*?\) ->.*")
                for line in docstr.split("\n")[2: -1]:
                    if re.match(funcsig_reg, line):
                        continue
                    output.append('    {docline}'.format(docline=line))
            output.append("    \"\"\"")
            output.append("    ...\n")


def generate_c_type_stub_custom(module: ModuleType,
                                class_name: str,
                                obj: type,
                                output: List[str],
                                imports: List[str],
                                sigs: Optional[Dict[str, str]] = None,
                                class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for a single class using runtime introspection.
       (Custom modified version, prototype is mypy.stubgenc.generate_c_type_stub)

    The result lines will be appended to 'output'. If necessary, any
    required names will be added to 'imports'.
    """
    # typeshed gives obj.__dict__ the not quite correct type Dict[str, Any]
    # (it could be a mappingproxy!), which makes mypyc mad, so obfuscate it.
    obj_dict: Mapping[str, Any] = getattr(obj, "__dict__")  # noqa
    items = sorted(obj_dict.items(), key=lambda x: method_name_sort_key(x[0]))
    methods: List[str] = []
    types: List[str] = []
    static_properties: List[str] = []
    rw_properties: List[str] = []
    ro_properties: List[str] = []
    done: Set[str] = set()
    for attr, value in items:
        if is_c_method(value) or is_c_classmethod(value):
            done.add(attr)
            if not is_skipped_attribute(attr):
                if attr == '__new__':
                    # TODO: We should support __new__.
                    if '__init__' in obj_dict:
                        # Avoid duplicate functions if both are present.
                        # But is there any case where .__new__() has a
                        # better signature than __init__() ?
                        continue
                    attr = '__init__'
                if is_c_classmethod(value):
                    methods.append('@classmethod')
                    self_var = 'cls'
                else:
                    self_var = 'self'
                generate_c_function_stub_costum(module, attr, value, methods, imports=imports,
                                                self_var=self_var, sigs=sigs, class_name=class_name,
                                                class_sigs=class_sigs)
        elif is_c_property(value):
            done.add(attr)
            generate_c_property_stub(attr, value, static_properties, rw_properties, ro_properties,
                                     is_c_property_readonly(value),
                                     module=module, imports=imports)
        elif is_c_type(value):
            generate_c_type_stub_custom(module, attr, value, types, imports=imports, sigs=sigs,
                                        class_sigs=class_sigs)
            done.add(attr)

    for attr, value in items:
        if is_skipped_attribute(attr):
            continue
        if attr not in done:
            static_properties.append('%s: ClassVar[%s] = ...' % (
                attr, strip_or_import(get_type_fullname(type(value)), module, imports)))
    all_bases = type.mro(obj)
    if all_bases[-1] is object:
        # TODO: Is this always object?
        del all_bases[-1]
    # remove pybind11_object. All classes generated by pybind11 have pybind11_object in their MRO,
    # which only overrides a few functions in object type
    if all_bases and all_bases[-1].__name__ == 'pybind11_object':
        del all_bases[-1]
    # remove the class itself
    all_bases = all_bases[1:]
    # Remove base classes of other bases as redundant.
    bases: List[type] = []
    for base in all_bases:
        if not any(issubclass(b, base) for b in bases):
            bases.append(base)
    if bases:
        bases_str = '(%s)' % ', '.join(
            strip_or_import(
                get_type_fullname(base),
                module,
                imports
            ) for base in bases
        )
    else:
        bases_str = ''
    if types or static_properties or rw_properties or methods or ro_properties:
        output.append('class %s%s:' % (class_name, bases_str))
        # append class comment
        output.append('    \"\"\"')
        docstr = getattr(obj, '__doc__', None)
        for line in (docstr.split('\n') if docstr else []):
            output.append('    {line}'.format(line=line))
        output.append('    \"\"\"')

        for line in types:
            if output and output[-1] and \
                    not output[-1].startswith('class') and line.startswith('class'):
                output.append('')
            output.append('    ' + line)
        for line in static_properties:
            output.append('    %s' % line)
        for line in rw_properties:
            output.append('    %s' % line)
        for line in methods:
            output.append('    %s' % line)
        for line in ro_properties:
            output.append('    %s' % line)
    else:
        output.append('class %s%s:' % (class_name, bases_str))
        # append class comment
        output.append('    \"\"\"')
        docstr = getattr(obj, '__doc__', None)
        for line in (docstr.split('\n') if docstr else []):
            output.append('    {line}'.format(line=line))
        output.append('    \"\"\"')
        output.append('    ...')
