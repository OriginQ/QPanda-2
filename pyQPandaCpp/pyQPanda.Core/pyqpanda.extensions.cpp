#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "QPandaConfig.h"
#include "QPanda.h"
#include "Extensions/Extensions.h"


using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;


USING_QPANDA

void init_extension_class(py::module & m)
{
#ifdef USE_EXTENSION
    py::class_<HHLAlg>(m, "HHLAlg")
        .def(py::init<QuantumMachine*>())
        .def("get_hhl_circuit",
            [](HHLAlg& self, const QStat& A, const std::vector<double>& b, const uint32_t& precision_cnt) {
                return self.get_hhl_circuit(A, b, precision_cnt); },
            "matrix_A"_a, "data_b"_a, "precision_cnt"_a = 0,
            py::return_value_policy::automatic)
        .def("check_QPE_result", &HHLAlg::check_QPE_result, "check QPE result")
        .def("get_amplification_factor", &HHLAlg::get_amplification_factor, "get_amplification_factor")
        .def("get_ancillary_qubit", &HHLAlg::get_ancillary_qubit, "get_ancillary_qubit")
        .def("get_qubit_for_b", [](HHLAlg& self) {
            std::vector<Qubit *> q_vec = self.get_qubit_for_b();
            return q_vec; },
            "get_qubit_for_b", py::return_value_policy::reference)
        .def("get_qubit_for_QFT", [](HHLAlg& self) {
            std::vector<Qubit *> q_vec = self.get_qubit_for_QFT();
            return q_vec; },
            "get_qubit_for_QFT", py::return_value_policy::reference)
        .def("query_uesed_qubit_num", &HHLAlg::query_uesed_qubit_num, "query_uesed_qubit_num")
        ;
#endif
    return ;
}

void init_extension_funtion(py::module & m)
{
#ifdef USE_EXTENSION
    m.def("expand_linear_equations", [](QStat& A, std::vector<double>& b) {
        HHLAlg::expand_linear_equations(A, b);

        py::list linear_equations_data;
        linear_equations_data.append(A);
        linear_equations_data.append(b);
        return linear_equations_data;
    }
        , "matrix"_a, "list"_a,
        "/**\
        * @brief  Extending linear equations to N dimension, N = 2 ^ n\
        * @ingroup QAlg\
        * @param[in] QStat& the source matrix, which will be extend to N*N, N = 2 ^ n\
        * @param[in] std::vector<double>& the source vector b, which will be extend to 2 ^ n\
        * @return\
        * @note\
        * / ",
        py::return_value_policy::automatic_reference
        );

    m.def("build_HHL_circuit", [](const QStat& A, const std::vector<double>& b, QuantumMachine *qvm, uint32_t precision_cnt /*= 0*/) {
        return build_HHL_circuit(A, b, qvm, precision_cnt);
    }, "/**\
        * @brief  build the quantum circuit for HHL algorithm to solve the target linear systems of equations : Ax = b\
        * @ingroup QAlg\
        * @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N = 2 ^ n\
        * @param[in] std::vector<double>& a given vector\
        * @param[in] uint32_t The count of digits after the decimal point,\
        default is 0, indicates that there are only integer solutions\
        * @return  QCircuit The whole quantum circuit for HHL algorithm\
        * @note The higher the precision is, the more qubit number and circuit - depth will be,\
        for example: 1 - bit precision, 4 additional qubits are required,\
            for 2 - bit precision, we need 7 additional qubits, and so on.\
            The final solution = (HHL result) * (normalization factor for b) * (1 << ceil(log2(pow(10, precision_cnt))))\
                * / ",
        "matrix_A"_a, "data_b"_a, "QuantumMachine"_a, "precision_cnt"_a = 0,
        py::return_value_policy::automatic
    );

    m.def("HHL_solve_linear_equations", [](const QStat& A, const std::vector<double>& b, uint32_t precision_cnt/* = 0*/) {
        return HHL_solve_linear_equations(A, b, precision_cnt);
    }, "/**\
        * @brief  Use HHL algorithm to solve the target linear systems of equations : Ax = b\
        * @ingroup QAlg\
        * @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N = 2 ^ n\
        * @param[in] std::vector<double>& a given vector\
        * @param[in] uint32_t The count of digits after the decimal point,\
        default is 0, indicates that there are only integer solutions.\
        * @return  QStat The solution of equation, i.e.x for Ax = b\
        * @note The higher the precision is, the more qubit number and circuit - depth will be,\
        for example: 1 - bit precision, 4 additional qubits are required,\
            for 2 - bit precision, we need 7 additional qubits, and so on.\
                * / ",
        "A"_a, "b"_a, "precision_cnt"_a = 0,
        py::return_value_policy::automatic
    );

    m.def("OBMT_mapping", [](QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
        bool optimization = false,
        uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
        uint32_t max_children = (std::numeric_limits<uint32_t>::max)(),
        const std::string& config_data = CONFIG_PATH) {
        QVec qv;
        auto ret_prog = OBMT_mapping(prog, quantum_machine, qv, optimization, max_partial, max_children, config_data);
        return ret_prog;
    }, "prog"_a, "quantum_machine"_a, "b_optimization"_a=false,
        "max_partial"_a= (std::numeric_limits<uint32_t>::max)(),
        "max_children"_a = (std::numeric_limits<uint32_t>::max)(),
        "config_data"_a = CONFIG_PATH,
        "/**\
        * @brief OPT_BMT mapping\
        * @ingroup Utilities\
        * @param[in] prog  the target prog\
        * @param[in] QuantumMachine *  quantum machine\
        * @param[in] uint32_t  Limits the max number of partial solutions per step, There is no limit by default\
        * @param[in] uint32_t  Limits the max number of candidate - solutions per double gate, There is no limit by default\
        * @param[in] const std::string config data, @See JsonConfigParam::load_config()\
        * @return QProg   mapped  quantum program\
        * / ",
        py::return_value_policy::automatic
        );

    m.def("virtual_z_transform", [](QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
        bool b_del_rz_gate = false,
        const std::string& config_data = CONFIG_PATH) {
        return virtual_z_transform(prog, quantum_machine, b_del_rz_gate, config_data);;
        }, "prog"_a, "quantum_machine"_a, "b_del_rz_gate"_a = false, "config_data"_a = CONFIG_PATH,
        py::return_value_policy::automatic
        );

    m.def("matrix_decompose", [](QVec& qubits, QStat& src_mat, const DecompositionMode mode, bool b_positive_seq) {
        switch (mode) {
        case DecompositionMode::QR:
            return matrix_decompose_qr(qubits, src_mat, b_positive_seq);
            //break;
        case DecompositionMode::HOUSEHOLDER_QR:
            return matrix_decompose_householder(qubits, src_mat, b_positive_seq);
            //break;
        default:
            throw std::runtime_error("Error: DecompositionMode");
        }
    }, "qubits"_a, "matrix"_a, "mode"_a = DecompositionMode::QR, "b_positive_seq"_a = true,
        "/**\
        * @brief  matrix decomposition\
        * @ingroup Utilities\
        * @param[in]  QVec& the used qubits\
        * @param[in]  QStat& The target matrix\
        * @param[in]  DecompositionMode decomposition mode, default is QR\
		* @param[in] const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), \
		             default is true\
        * @return    QCircuit The quantum circuit for target matrix\
        * @see DecompositionMode\
        * / ",
        py::return_value_policy::automatic
        );

    m.def("matrix_decompose_pualis", [](QuantumMachine* qvm, const EigenMatrixX& mat
        ) {
            PualiOperatorLinearCombination linearcom;
            matrix_decompose_pualis(qvm, mat, linearcom);
            return linearcom;
        }, 
            py::return_value_policy::automatic
            );
#else
    m.def("matrix_decompose", [](QVec& qubits, QStat& src_mat, const DecompositionMode mode, bool b_positive_seq) {
        switch (mode) {
        case DecompositionMode::QR:
            return matrix_decompose_qr(qubits, src_mat, b_positive_seq);
            break;
        case DecompositionMode::HOUSEHOLDER_QR:
            throw std::runtime_error("Error: only QPanda extensions support.");
            break;
        default:
            throw std::runtime_error("Error: DecompositionMode");
        }
    }, "qubits"_a, "matrix"_a, "mode"_a = DecompositionMode::QR, "b_positive_seq"_a = true,
        "/**\
        * @brief  matrix decomposition\
        * @ingroup Utilities\
        * @param[in]  QVec& the used qubits\
        * @param[in]  QStat& The target matrix\
        * @param[in]  DecompositionMode decomposition mode, default is QR\
        * @param[in] const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), \
		             default is true\
        * @return    QCircuit The quantum circuit for target matrix\
        * @see DecompositionMode\
        * / ",
        py::return_value_policy::automatic
        );
#endif
    return ;
}

