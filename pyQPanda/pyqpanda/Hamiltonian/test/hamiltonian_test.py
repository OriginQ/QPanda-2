from pyqpanda.Hamiltonian import PauliOperator
def hamiltonian_operator_overload():
    op_dict={'X2 z1 y3 x4':5.0, "z0 X2 y5":8}
    op_dict2={'X2 z0 y3 x4 y1':-5}

    m1=PauliOperator(op_dict)
    m2=PauliOperator(op_dict2)
    print(m1)
    print(m2)
    m1+=m2
    m1.arrange()
    print(m1)

    