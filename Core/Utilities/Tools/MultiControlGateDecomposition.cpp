#include <bitset>
#include <numeric>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include <EigenUnsupported/Eigen/KroneckerProduct>
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

USING_QPANDA

using namespace Eigen;
using  qmatrix_t = Matrix<qcomplex_t, Dynamic, Dynamic, RowMajor>;

QGate root_matrix(Qubit* tar_qubit, Qubit* ctr_qubit, QStat unitary, int root)
{
    qmatrix_t matrix = qmatrix_t::Map(&unitary[0], 2, 2);

    Eigen::ComplexEigenSolver<qmatrix_t> solver(matrix);
    auto evecs = solver.eigenvectors();
    auto evals = solver.eigenvalues();

    for (auto i = 0; i < evals.size(); ++i)
        evals[i] = std::pow(evals[i], (double)(1. / root));

    auto root_unitary = evecs * evals.asDiagonal() * evecs.adjoint();

    unitary.clear();
    for (size_t i = 0; i < root_unitary.rows(); ++i)
    {
        for (size_t j = 0; j < root_unitary.cols(); ++j)
        {
            unitary.emplace_back((qcomplex_t)(root_unitary(i, j)));
        }
    }

    return U4(unitary, tar_qubit).control({ ctr_qubit });
}

QCircuit LinearDepthDecomposition::PnRx(QVec qubits, QStat matrix)
{
    auto n = qubits.size() - 1;

    //k = 2 ~ n
    QCircuit circuit;
    for (auto k = 1; k < n; ++k)
    {
        circuit << RX(qubits[n], PI / (1ull << (n - k))).control({ qubits[k] });
    }

    return circuit;
}

QCircuit LinearDepthDecomposition::PnU(QVec qubits, QStat matrix)
{
    auto n = qubits.size() - 1;

    QCircuit circuit;
    for (auto k = 1; k < n; ++k)
    {
        circuit << root_matrix(qubits[n], qubits[k], matrix, 1ull << (n - k));
    }

    return circuit;
}

QCircuit LinearDepthDecomposition::CnRx(QVec qubits, QStat matrix)
{
    QCircuit circuit;
    circuit << Qn(qubits, matrix).dagger();
    circuit << PnRx(qubits, matrix).dagger();
    circuit << Qn(qubits, matrix);
    circuit << RX(qubits.back(), PI / (1ull << (qubits.size() - 2))).control({ qubits.front() });

    circuit << PnRx(qubits, matrix);
    return circuit;
}

QCircuit LinearDepthDecomposition::Qn(QVec qubits, QStat matrix)
{
    auto n = qubits.size() - 1;

    QCircuit circuit;
    for (auto i = 0; i < n - 1; ++i)
    {
        QVec temp_qubits(qubits.begin(), qubits.begin() + i + 2);
        circuit << CnRx(temp_qubits, matrix);
    }

    return circuit;
}

QCircuit LinearDepthDecomposition::CnU(QVec qubits, QStat matrix)
{
    QCircuit circuit;
    circuit << Qn(qubits, matrix).dagger();
    circuit << PnU(qubits, matrix).dagger();
    circuit << Qn(qubits, matrix);
    circuit << root_matrix(qubits.back(), qubits.front(), matrix, 1ull << (qubits.size() - 2));

    circuit << PnU(qubits, matrix);
    return circuit;
}

void LinearDepthDecomposition::execute(std::shared_ptr<QNode> node, std::shared_ptr<QNode> parent_node)
{
    switch (node->getNodeType())
    {
    case GATE_NODE:
        return execute(std::dynamic_pointer_cast<AbstractQGateNode>(node), parent_node);
        break;

    case CIRCUIT_NODE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractQuantumCircuit>(node), parent_node);
        break;

    case PROG_NODE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractQuantumProgram>(node), parent_node);
        break;

    case MEASURE_GATE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractQuantumMeasure>(node), parent_node);
        break;

    case CLASS_COND_NODE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractClassicalProg>(node), parent_node);
        break;

    case QIF_START_NODE:
    case WHILE_START_NODE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractControlFlowNode>(node), parent_node);
        break;

    case RESET_NODE:
        return QNodeDeepCopy::execute(std::dynamic_pointer_cast<AbstractQuantumReset>(node), parent_node);
        break;
    }

    QCERR_AND_THROW(run_fail, "error: unsupport copy-node type.");
}

static QStat get_U4_matrix(prob_vec params, bool is_dagger)
{
    QPANDA_ASSERT(4 != params.size(), "U4 params error");

    auto alpha = params[0];
    auto beta = params[1];
    auto gamma = params[2];
    auto delta = params[3];

    QStat _U4;

    QStat matrix;
    _U4.emplace_back(qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2)));
    _U4.emplace_back(qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2)));
    _U4.emplace_back(qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2)));
    _U4.emplace_back(qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2)));

    if (is_dagger) dagger(_U4);
    return _U4;
}

void LinearDepthDecomposition::execute(std::shared_ptr<AbstractQGateNode> cur_node, 
    std::shared_ptr<QNode> parent_node)
{
    QVec qvec;
    cur_node->getQuBitVector(qvec);

    QVec controls;
    cur_node->getControlVector(controls);

    Qnum control_qubits_addr;
    for (auto val : controls)
    {
        control_qubits_addr.emplace_back(val->get_phy_addr());
    }

    std::stable_sort(control_qubits_addr.begin(), control_qubits_addr.end());
    if (std::adjacent_find(control_qubits_addr.begin(), control_qubits_addr.end()) != control_qubits_addr.end())
        QCERR_AND_THROW(run_fail, "duplicate control qubit");

    auto gate_type = (GateType)cur_node->getQGate()->getGateType();

    if (!controls.size())
        return QNodeDeepCopy::execute(cur_node, parent_node);

    if (is_single_gate(gate_type) && 1 == controls.size())
        return QNodeDeepCopy::execute(cur_node, parent_node);

    switch (gate_type)
    {
    case GateType::PAULI_X_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::HADAMARD_GATE:
    case GateType::T_GATE:
    case GateType::S_GATE:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    case GateType::U1_GATE:
    case GateType::U2_GATE:
    case GateType::U3_GATE:
    case GateType::U4_GATE:
    case GateType::RPHI_GATE:
    {
        QStat matrix;
        cur_node->getQGate()->getMatrix(matrix);

        if (cur_node->isDagger()) dagger(matrix);

        auto circuit = CnU(controls + qvec, matrix);
        insert(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), parent_node);
        break;
    }

    case GateType::SWAP_GATE:
    {
        // SWAP(0, 1) => CNOT(0, 1) + CNOT(1, 0) + CNOT(0, 1)
        QStat _X = { 0, 1, 1, 0 };

        QCircuit result_circuit;
        result_circuit <<  CnU(controls + QVec({ qvec[0],qvec[1] }), _X);
        result_circuit <<  CnU(controls + QVec({ qvec[1],qvec[0] }), _X);
        result_circuit <<  CnU(controls + QVec({ qvec[0],qvec[1] }), _X);

        result_circuit.setDagger(cur_node->isDagger());

        insert(std::dynamic_pointer_cast<QNode>(result_circuit.getImplementationPtr()), parent_node);
        break;
    }

    case GateType::ISWAP_GATE:
    {
        // iSWAP(0, 1) => 
        // CU(0, 1)(1.570796, 3.141593, 0.000000, 0.000000).dag + 
        // CU(0, 1)(-1.570796, 6.283185, 3.141593, 0.000000).dag +
        // CU(1, 0)(-1.570796, 3.141593, 3.141593, 0.00000).dag +
        // CU(0, 1)(1.570796, 6.283185, 3.141593, 0.00000).dag

        prob_vec params1 = { PI / 2, PI, 0, 0 };
        prob_vec params2 = { -PI / 2, 2 * PI, PI, 0 };
        prob_vec params3 = { -PI / 2, PI, PI, 0 };
        prob_vec params4 = { PI / 2, 2 * PI, PI, 0 };

        QCircuit result_circuit;

        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params1, true));
        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params2, true));
        result_circuit << CnU(controls + QVec({ qvec[1],qvec[0] }), get_U4_matrix(params3, true));
        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params4, true));

        result_circuit.setDagger(cur_node->isDagger());

        insert(std::dynamic_pointer_cast<QNode>(result_circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::SQISWAP_GATE:
    {
        // SqiSWAP(0, 1) => 
        // CU(0, 1)(1.570796, 3.141593, 0.000000, 0.000000).dag + 
        // CU(0, 1)(1.570796, 6.283185, 3.141593, 0.000000).dag +
        // CU(1, 0)(1.570796, 0.000000, 1.570796, 3.141593).dag +
        // CU(0, 1)(-1.570796, 6.283185, 3.141593, 0.00000).dag

        prob_vec params1 = { PI / 2, PI, 0, 0 };
        prob_vec params2 = { PI / 2, 2 * PI, PI, 0 };
        prob_vec params3 = { PI / 2, 0, PI / 2, PI };
        prob_vec params4 = { -PI / 2, 2 * PI, PI, 0 };

        QCircuit result_circuit;

        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params1, true));
        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params2, true));
        result_circuit << CnU(controls + QVec({ qvec[1],qvec[0] }), get_U4_matrix(params3, true));
        result_circuit << CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params4, true));

        result_circuit.setDagger(cur_node->isDagger());
        insert(std::dynamic_pointer_cast<QNode>(result_circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::CNOT_GATE:
    {
        QStat _X = { 0, 1, 1, 0 };

        auto circuit = CnU(controls + QVec({ qvec[0],qvec[1] }), _X);
        insert(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::CZ_GATE:
    {
        QStat _Z = { 1, 0, 0, -1 };

        auto circuit = CnU(controls + QVec({ qvec[0],qvec[1] }), _Z);
        insert(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::CPHASE_GATE:
    {
        auto param_ptr = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(cur_node->getQGate());
        auto param = param_ptr->getParameter();

        QStat _U1 = { 1, 0, 0, std::exp(qcomplex_t(0, param)) };

        auto circuit = CnU(controls + QVec({ qvec[0],qvec[1] }), _U1);

        circuit.setDagger(cur_node->isDagger());
        insert(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::CU_GATE:
    {
        auto param_ptr = dynamic_cast<QGATE_SPACE::AbstractAngleParameter *>(cur_node->getQGate());

        prob_vec params;
        params.emplace_back(param_ptr->getAlpha());
        params.emplace_back(param_ptr->getBeta());
        params.emplace_back(param_ptr->getGamma());
        params.emplace_back(param_ptr->getDelta());

        auto circuit = CnU(controls + QVec({ qvec[0],qvec[1] }), get_U4_matrix(params, false));
        circuit.setDagger(cur_node->isDagger());

        insert(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), parent_node);
        break;
    }
    case GateType::MS_GATE:
    case GateType::RYY_GATE:
    case GateType::RXX_GATE:
    case GateType::RZZ_GATE:
    case GateType::RZX_GATE:
    case GateType::ORACLE_GATE:
    default:
    {
        QStat matrix;
        cur_node->getQGate()->getMatrix(matrix);

        if (cur_node->isDagger()) dagger(matrix);

        auto decomposed_circuit = matrix_decompose_qr(qvec, matrix, false);

        decomposed_circuit.setControl(controls);

        QProg node(decomposed_circuit);
        flatten(node);

        execute(std::dynamic_pointer_cast<QNode>(node.getImplementationPtr()),
            parent_node);

        break;
    }

    case GateType::I_GATE:
    case GateType::BARRIER_GATE: break;
    }
}

