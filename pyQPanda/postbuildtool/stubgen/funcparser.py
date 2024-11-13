import regfilter
import parso

def _matchVarName(node, name: str):
    """
    Recursively searches for a node with a matching variable name within a quantum circuit
    representation. This function is designed to traverse a quantum circuit's abstract syntax tree
    (AST) and check if any node within the tree has a type of 'name' and a value that matches the
    specified name.

    Parameters:
    - node: An instance representing a node in the quantum circuit AST.
    - name: A string representing the variable name to be matched.

    Returns:
    - A boolean indicating whether a node with the specified name is found.

    This function is intended for use within the pyQPanda package, which facilitates the
    programming of quantum computers using quantum circuits and gates, operating on a quantum
    circuit simulator or quantum cloud service.
    """
    if node.type == "name" and node.value == name:
        return True
    elif hasattr(node, "children"):
        for child in node.children:
            return _matchVarName(child, name)
    else:
        return False


def _getFuncVarList(node):
    """
    Recursively extracts the variable list from a given `parso.python.tree.Node` object.
    
    If the node is an instance of `parso.python.tree.PythonErrorNode`, the function raises an exception.
    
    For nodes representing function parameters, it returns the list of variables from the second to the second-last child.
    
    If the node has children, the function traverses them recursively. It returns the first non-empty list of variables found.
    
    If no variables are found, the function returns `None`.
    
    Parameters:
    node (parso.python.tree.Node): The node to analyze for variable information.
    
    Returns:
    list: A list of variables found within the node, or `None` if none are found.
    
    Raises:
    Exception: If the node is an instance of `parso.python.tree.PythonErrorNode`.
    """
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
    """
    Retrieves the string representation of a specified variable name from a function declaration.

    This function parses the provided function declaration using the `parso` library to construct an Abstract Syntax Tree (AST).
    It then traverses the AST to locate the variable with the given name. If the variable is found, the function extracts and returns
    the string slice of the function declaration corresponding to that variable.

    Parameters:
    - func_decl (str): A string containing the function declaration from which to extract the variable.
    - var_name (str): The name of the variable to find within the function declaration.

    Returns:
    - str: The string slice of the function declaration corresponding to the variable.

    Raises:
    - Exception: If the variable is not found in the function declaration or if there is an error during parsing or variable extraction.

    The function is intended for internal use within the pyQPanda package, which is designed for programming quantum computers using quantum circuits and gates.
    It is located in the 'pyQPanda.postbuildtool.stubgen.funcparser.py' directory.
    """
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
