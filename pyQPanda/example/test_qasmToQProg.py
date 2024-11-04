

import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

#from pyqpanda import *

from pyqpanda.pyQPanda import QMachineType, convert_qasm_string_to_qprog, destroy_quantum_machine, init_quantum_machine
from pyqpanda.pyQPanda import convert_qasm_to_qprog
import unittest

class Test_QASMToQProg(unittest.TestCase):
    qasm_str_list = []
    SX_qasm = """OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        creg c[2];
        SX q[1];
        sx q[0];
    """
    qasm_str_list.append(SX_qasm)
    SXdg_qasm ="""
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        creg c[2];
        SXdg q[1];
        sxdg q[0];
    """
    qasm_str_list.append(SXdg_qasm)
    ISWAP_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        iswap q[0],q[1];
    """
    qasm_str_list.append(ISWAP_qasm)
    DCX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        DCX q[0],q[1];
        dcx q[0],q[1];
    """
    qasm_str_list.append(DCX_qasm)
    CP_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        CP(-PI/2) q[0],q[1];
        cp(-PI/2) q[0],q[1];
    """
    qasm_str_list.append(CP_qasm)
    CS_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        CS q[0],q[1];
        cs q[0],q[1];
    """
    qasm_str_list.append(CS_qasm)
    CSdg_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        CSdg q[0],q[1];
        csdg q[0],q[1];
    """
    qasm_str_list.append(CSdg_qasm)
    CCZ_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        CCZ q[0],q[1],q[2];
        ccz q[0],q[1],q[2];
    """
    qasm_str_list.append(CCZ_qasm)
    ECR_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        ECR q[0],q[1];
        ecr q[0],q[1];
    """
    qasm_str_list.append(ECR_qasm)
    R_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        R(3.14,3.15) q[0];
        r(3.14,3.15) q[0];
    """
    qasm_str_list.append(R_qasm)
    XXMinusYY_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        XXMinusYY(3.14,3.15) q[0],q[1];
        xx_minus_yy(3.14,3.15) q[0],q[1];
    """
    qasm_str_list.append(XXMinusYY_qasm)
    XXPlusYY_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        XXPlusYY(3.14,3.15) q[0],q[1];
        xx_plus_yy(3.14,3.15) q[0],q[1];
    """
    qasm_str_list.append(XXPlusYY_qasm)
    V_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        V q[0];
        v q[0];
    """
    qasm_str_list.append(V_qasm)
    W_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        W q[0];
        w q[0];
    """
    qasm_str_list.append(W_qasm)

    CCX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        toffoli q[0], q[1], q[2];
        TOFFOLI q[0], q[1], q[2];
        ccx q[0], q[1], q[2];
        CCX q[0], q[1], q[2];
    """
    qasm_str_list.append(CCX_qasm)
    CH_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        ch q[0], q[1];
        CH q[0], q[1];
    """
    qasm_str_list.append(CH_qasm)
    CNOT_CZ_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        cx q[0], q[1];
        CX q[0], q[1];
        cnot q[0], q[1];
        CNOT q[0], q[1];

        cz q[0], q[1];
        CZ q[1], q[0];
    """
    qasm_str_list.append(CNOT_CZ_qasm)
    CRX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[2] q;
        crx(3.14) q[0], q[1];
        CRX(3.14) q[0], q[1];
    """
    qasm_str_list.append(CRX_qasm)
    CRY_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[2] q;
        cry(3.14) q[0], q[1];
        CRY(3.14) q[0], q[1];
    """
    qasm_str_list.append(CRY_qasm)
    CRZ_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[2] q;
        crz(3.14) q[0], q[1];
        CRZ(3.14) q[0], q[1];
    """
    qasm_str_list.append(CRZ_qasm)
    CSWAP_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[3] q;

        cswap q[0], q[1], q[2];
        CSWAP q[0], q[1], q[2];
    """
    qasm_str_list.append(CSWAP_qasm)
    CSX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        csx q[0], q[1];
        CSX q[0], q[1];
    """
    qasm_str_list.append(CSX_qasm)
    CU_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        cu(3.14, 3.15, 3.16, 3.17) q[1], q[2];
        CU(3.14, 3.15, 3.16, 3.17) q[1], q[2];
    """
    qasm_str_list.append(CU_qasm)
    CU1_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        cp(3.14)q[0], q[1];
        cu1(3.14)q[0], q[1];
        CP(3.14)q[0], q[1];
        CU1(3.14)q[0], q[1];
        cphase(3.14)q[0], q[1];
        CPHASE(3.14)q[0], q[1];
    """
    qasm_str_list.append(CU1_qasm)
    CU3_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        cu3(3.14, 3.15, 3.16) q[1], q[2];
        CU3(3.14, 3.15, 3.16) q[1], q[2];
    """
    qasm_str_list.append(CU3_qasm)
    CY_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        cy q[0], q[1];
        CY q[0], q[1];
    """
    qasm_str_list.append(CY_qasm)
    C3SQRTX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        c3sqrtx q[0], q[1], q[2], q[3];
        C3SQRTX q[0], q[1], q[2], q[3];
    """
    qasm_str_list.append(C3SQRTX_qasm)
    C3X_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        c3x q[0], q[1], q[2], q[3];
        C3X q[0], q[1], q[2], q[3];
    """
    qasm_str_list.append(C3X_qasm)
    C4X_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[5] q;

        c4x q[0], q[1], q[2], q[3], q[4];
        C4X q[0], q[1], q[2], q[3], q[4];
    """
    qasm_str_list.append(C4X_qasm)
    H_I_X_Y_Z_S_T_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;
        h q[0];
        H q[1];

        i q[0];
        id q[1];
        I q[2];
        u0 q[3];
         
        x q[0];
        X q[1];

        y q[0];
        Y q[1];

        z q[0];
        Z q[1];

        s q[0];
        S q[1];

        t q[0];
        T q[1];
    """
    qasm_str_list.append(H_I_X_Y_Z_S_T_qasm)
    RCCX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[3] q;
        rccx q[0], q[1], q[2];
        RCCX q[0], q[1], q[2];
    """
    qasm_str_list.append(RCCX_qasm)

    RC3X_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;
        rc3x q[0], q[1], q[2], q[3];
        RC3X q[0], q[1], q[2], q[3];
    """
    qasm_str_list.append(RC3X_qasm)
    RXX_RYY_RZZ_RZX_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        rxx(3.14) q[0], q[1];
        RXX(3.15) q[0], q[1];

        ryy(3.14) q[0], q[1];
        RYY(3.15) q[0], q[1];

        rzz(3.14) q[0], q[1];
        RZZ(3.15) q[0], q[1];

        rzx(3.14) q[0], q[1];
        RZX(3.15) q[0], q[1];
    """
    qasm_str_list.append(RXX_RYY_RZZ_RZX_qasm)
    RX_RY_RZ_P_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        rx(3.14) q[0];
        RX(3.15) q[1];

        ry(3.14) q[0];
        RY(3.15) q[1];

        rz(3.14) q[0];
        RZ(3.15) q[1];

        p(3.14) q[0];
        P(3.15) q[0];
        u1(3.16) q[0];
        U1(3.17) q[0];
        phase(3.18) q[0];
    """
    qasm_str_list.append(RX_RY_RZ_P_qasm)
    SDG_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[2] q;

        sdg q[0];
        SDG q[0];
        Sdg q[0];
    """
    qasm_str_list.append(SDG_qasm)
    SWAP_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        swap q[0], q[1];
        SWAP q[1], q[0];
    """
    qasm_str_list.append(SWAP_qasm)

    TDG_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[2] q;

        tdg q[0];
        TDG q[0];
        Tdg q[0];
    """
    qasm_str_list.append(TDG_qasm)
    U2_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        u2(3.14, 3.15) q[0];
        U2(3.14, 3.15) q[0];
    """
    qasm_str_list.append(U2_qasm)
    U3_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[4] q;

        u(3.14, 3.15, 3.16) q[0];
        u3(3.14, 3.15, 3.16) q[0];
        U(3.14, 3.15, 3.16) q[0];
        U3(3.14, 3.15, 3.16) q[0];
    """
    qasm_str_list.append(U3_qasm)
    

    RESET_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[3] q;
        reset q[0];
    """
    qasm_str_list.append(RESET_qasm)
    MEASURE_qasm = """
        qubit[4] q;
        bit[4] c;
        x q[2];
        c[2] = measure q[2];
    """
    qasm_str_list.append(MEASURE_qasm)
    BARRIER_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[5] q;
        barrier q[0],q[1],q[2],q[3];
    """
    qasm_str_list.append(BARRIER_qasm)
    Classical_expr_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";

        qubit[5] q;
        bit[2] c;
        rz(pi - 5) q[0];
        c[1] = measure q[0]; 
    """
    qasm_str_list.append(Classical_expr_qasm)
    
    def test_qasm_to_qprog(self,qasm_str):
        print("### qasm_str:",qasm_str)
        machine = init_quantum_machine(QMachineType.CPU)
        qprog,qbits,cbits = convert_qasm_string_to_qprog(qasm_str,machine)
        print("qprog:",end='\n')
        print(qprog,end='\n')
        print("qbits:",end='\n')
        print(qbits)
        print("cbits:",end='\n')
        print(cbits)
        
        destroy_quantum_machine(machine)


if __name__=="__main__":
    #unittest.main(verbosity=2)
    TQL = Test_QASMToQProg()
    for qasm_str in TQL.qasm_str_list:
        TQL.test_qasm_to_qprog(TQL.U2_qasm)
    

