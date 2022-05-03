import regfilter
import parso

def _matchVarName(node, name: str):
    if node.type == "name" and node.value == name:
        return True
    elif hasattr(node, "children"):
        for child in node.children:
            return _matchVarName(child, name)
    else:
        return False


def _getFuncVarList(node):
    if isinstance(node, parso.python.tree.PythonErrorNode):
        raise

    if node.type == "parameters":
        vallist = node.children[1:-1]
        return vallist
    elif hasattr(node, "children"):
        for child in node.children:
            varlist = _getFuncVarList(child)
            if varlist:
                return varlist
    else:
        return None


def getFuncVarStr(func_decl, var_name: str) -> str:
    grammar = parso.load_grammar()
    ast = grammar.parse(func_decl)

    varlist = []
    try:
        varlist = _getFuncVarList(ast)
    except:
        func_decl = regfilter.filterFunc(func_decl)
        ast = grammar.parse(func_decl)
        try:
            varlist = _getFuncVarList(ast)
        except:
            raise Exception("some error in "+ func_decl)

    var_str = ""
    for var_node in varlist:
        assert(var_node.type == "param")
        for child in var_node.children:
            if _matchVarName(child, var_name):
                start_pos = var_node.start_pos[1]
                end_pos = var_node.end_pos[1]
                var_str = func_decl[start_pos: end_pos]
                return var_str


if __name__ == "__main__":
    s = 'def set_measure_error(self: pyQPanda.NoiseQVM, model: pyQPanda.NoiseModel, prob: float = float, qubits: pyQPanda.QVec = ::sd) -> None:...'
    var_str = getFuncVarStr(s, 'qubits')
