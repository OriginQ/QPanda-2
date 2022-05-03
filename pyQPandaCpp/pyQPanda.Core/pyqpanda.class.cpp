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

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

void export_core_class(py::module &m)
{
    py::class_<NodeInfo>(m, "NodeInfo", "Detailed information of a QProg node")
        .def(py::init<>())
        .def(py::init<const NodeIter, QVec, QVec, int, const bool>(),
             py::arg("iter"),
             py::arg("target_qubits"),
             py::arg("control_qubits"),
             py::arg("type"),
             py::arg("dagger"))
        .def("reset", &NodeInfo::reset)
        .def_readwrite("m_iter", &NodeInfo::m_iter)
        .def_readwrite("m_node_type", &NodeInfo::m_node_type)
        .def_readwrite("m_gate_type", &NodeInfo::m_gate_type)
        .def_readwrite("m_is_dagger", &NodeInfo::m_is_dagger)
        .def_readwrite("m_target_qubits", &NodeInfo::m_target_qubits)
        .def_readwrite("m_control_qubits", &NodeInfo::m_control_qubits)
        .def_readwrite("m_cbits", &NodeInfo::m_cbits)
        .def_readwrite("m_params", &NodeInfo::m_params)
        .def_readwrite("m_name", &NodeInfo::m_name);

    py::class_<OriginCollection>(m, "OriginCollection", "A relatively free data collection class for saving data")
        .def(py::init<>())
        .def(py::init<std::string>(),
             py::arg("file_name"),
             "Construct a new Origin Collection by read a json file")
        .def(py::init<const OriginCollection &>())
        .def("setNames",
             [](OriginCollection &c, py::args args)
             {
                 std::vector<std::string> all_key;
                 for (auto arg : args)
                 {
                     all_key.push_back(std::string(py::str(arg)));
                 }
                 c = all_key;
             })
        .def(
            "insertValue",
            [](OriginCollection &c, std::string key, py::args args)
            {
                int i = 1;
                auto vector = c.getKeyVector();
                c.addValue(vector[0], key);
                for (auto arg : args)
                {
                    c.addValue(vector[i], std::string(py::str(arg)));
                    i++;
                }
            },
            py::arg("key"))
        .def("getValue",
             &OriginCollection::getValue,
             py::arg("key_name"),
             "Get value by Key name")
        .def("getValueByKey",
             py::overload_cast<const std::string &>(&OriginCollection::getValueByKey),
             py::arg("key_value"),
             "Get Value by key value")
        .def("getValueByKey",
             py::overload_cast<int>(&OriginCollection::getValueByKey),
             py::arg("key_value"),
             "Get Value by key value")
        .def("open",
             &OriginCollection::open,
             py::arg("file_name"),
             "Read the json file of the specified path")
        .def("write", &OriginCollection::write, "write json file")
        .def("getJsonString", &OriginCollection::getJsonString, "Get Json String")
        .def("getFilePath", &OriginCollection::getFilePath, "Get file path")
        .def("getKeyVector", &OriginCollection::getKeyVector, "Get key vector");

    py::class_<QResult>(m, "QResult", "QResult abstract class, this class contains the result of the quantum measurement")
        .def("getResultMap", &QResult::getResultMap, py::return_value_policy::reference);

    py::class_<ClassicalProg>(m, "ClassicalProg")
        .def(py::init<ClassicalCondition &>());

    py::class_<QGate>(m, "QGate")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (GATE_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
                    return QGate(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }))
        .def("dagger", &QGate::dagger)
        .def("control", &QGate::control,
             py::arg("control_qubits"),
             "Get a control quantumgate  base on current quantum gate node")
        .def("is_dagger", &QGate::isDagger)
        .def("set_dagger", &QGate::setDagger)
        .def("set_control", &QGate::setControl)
        .def("get_target_qubit_num", &QGate::getTargetQubitNum)
        .def("get_control_qubit_num", &QGate::getControlQubitNum)
        .def("get_qubits",
             &QGate::getQuBitVector,
             py::arg("qubits"),
             "Get qubit vector inside this quantum gate\n"
             "\n"
             "Args:\n"
             "    qvec: qubits output\n"
             "\n"
             "Returns:\n"
             "    int: size of qubits",
             py::return_value_policy::automatic)
        .def("get_control_qubits",
             &QGate::getControlVector,
             py::arg("control_qubits"),
             "Get control vector fron current quantum gate node\n"
             "\n"
             "Args:\n"
             "    qvec: control qubits output\n"
             "\n"
             "Returns:\n"
             "    int: size of control qubits",
             py::return_value_policy::automatic)
        .def("gate_type",
             [](QGate &qgate)
             {
                 return qgate.getQGate()->getGateType();
             })
        .def(
            "gate_matrix",
            [](QGate &qgate)
            {
                QStat matrix;
                qgate.getQGate()->getMatrix(matrix);
                return matrix;
            },
            py::return_value_policy::automatic);

    /* QIfProg and QWhileProg will use QProg type, so we declare QProg type here, define it's function later, like C++ Forward declarations */
    py::class_<QProg> PyQProgClass(m, "QProg", "Quantum program,can construct quantum circuit,data struct is linked list");

    py::class_<QIfProg>(m, "QIfProg")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (QIF_START_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(*iter);
                    return QIfProg(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }))
        .def(py::init<ClassicalCondition, QProg>(),
             py::arg("classical_cond"),
             py::arg("true_branch_qprog"))
        .def(py::init<ClassicalCondition, QProg, QProg>(),
             py::arg("classical_cond"),
             py::arg("true_branch_qprog"),
             py::arg("false_branch_qprog"))
        .def(
            "get_true_branch",
            [](QIfProg &self)
            {
                auto true_branch = self.getTrueBranch();
                if (!true_branch)
                {
                    QCERR("true branch is null");
                    throw runtime_error("true branch is null");
                }

                auto type = true_branch->getNodeType();
                if (PROG_NODE != type)
                {
                    QCERR("true branch node type error");
                    throw runtime_error("true branch node type error");
                }

                return QProg(true_branch);
            },
            py::return_value_policy::automatic)
        .def(
            "get_false_branch",
            [](QIfProg &self)
            {
                auto false_branch = self.getFalseBranch();
                if (!false_branch)
                {
                    return QProg();
                }

                auto type = false_branch->getNodeType();
                if (PROG_NODE != type)
                {
                    QCERR("false branch node type error");
                    throw runtime_error("true branch node type error");
                }

                return QProg(false_branch);
            },
            py::return_value_policy::automatic)
        .def("get_classical_condition", &QIfProg::getClassicalCondition, py::return_value_policy::automatic);

    py::class_<QWhileProg>(m, "QWhileProg")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (WHILE_START_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(*iter);
                    return QWhileProg(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }))
        .def(py::init<ClassicalCondition, QProg>())
        .def(
            "get_true_branch",
            [](QWhileProg &self)
            {
                auto true_branch = self.getTrueBranch();
                if (!true_branch)
                {
                    QCERR("true branch is null");
                    throw runtime_error("true branch is null");
                }

                auto type = true_branch->getNodeType();
                if (PROG_NODE != type)
                {
                    QCERR("true branch node type error");
                    throw runtime_error("true branch node type error");
                }
                return QProg(true_branch);
            },
            py::return_value_policy::automatic)
        .def("get_classical_condition", &QWhileProg::getClassicalCondition, py::return_value_policy::automatic);
    ;

    py::class_<QMeasure>(m, "QMeasure")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (MEASURE_GATE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*iter);
                    return QMeasure(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }));

    py::class_<QReset>(m, "QReset")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (RESET_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQuantumReset>(*iter);
                    return QReset(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }));

    py::class_<QCircuit>(m, "QCircuit")
        .def(py::init<>())
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (CIRCUIT_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(*iter);
                    return QCircuit(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }))

        .def("__lshift__", &QCircuit::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("__lshift__", &QCircuit::operator<<<QGate>, py::return_value_policy::reference)

        .def("is_empty", &QCircuit::is_empty, py::return_value_policy::automatic)

        .def("insert", &QCircuit::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("insert", &QCircuit::operator<<<QGate>, py::return_value_policy::reference)
        .def("dagger", &QCircuit::dagger, py::return_value_policy::automatic)
        .def("control", &QCircuit::control,
             py::arg("control_qubits"),
             py::return_value_policy::automatic)
        .def("set_dagger", &QCircuit::setDagger)
        .def("set_control", &QCircuit::setControl,
             py::arg("control_qubits"))
        .def("begin", &QCircuit::getFirstNodeIter, py::return_value_policy::reference)
        .def("end", &QCircuit::getEndNodeIter, py::return_value_policy::reference)
        .def("last", &QCircuit::getLastNodeIter, py::return_value_policy::reference)
        .def("head", &QCircuit::getHeadNodeIter, py::return_value_policy::reference)
        .def(
            "__str__",
            [](QCircuit &p)
            {
                return draw_qprog(p);
            },
            py::return_value_policy::reference);

    PyQProgClass.def(py::init<>())
        .def(py::init<QProg &>())
        .def(py::init<QCircuit &>())
        .def(py::init<QIfProg &>())
        .def(py::init<QWhileProg &>())
        .def(py::init<QGate &>())
        .def(py::init<QMeasure &>())
        .def(py::init<QReset &>())
        .def(py::init<ClassicalCondition &>())
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (PROG_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(*iter);
                    return QProg(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }))
        /* template function should be instance explicit */
        .def("__lshift__", &QProg::operator<<<QProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QGate>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QIfProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QWhileProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QMeasure>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QReset>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<ClassicalCondition>, py::return_value_policy::reference)

        .def("is_empty", &QProg::is_empty, py::return_value_policy::automatic_reference)
        .def("insert", &QProg::operator<<<QProg>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QGate>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QIfProg>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QWhileProg>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QMeasure>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QReset>, py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<ClassicalCondition>, py::return_value_policy::reference)

        .def("begin", &QProg::getFirstNodeIter, py::return_value_policy::reference)
        .def("end", &QProg::getEndNodeIter, py::return_value_policy::reference)
        .def("head", &QProg::getHeadNodeIter, py::return_value_policy::reference)
        .def("last", &QProg::getLastNodeIter, py::return_value_policy::reference)
        .def(
            "__str__",
            [](QProg &p)
            {
                return draw_qprog(p);
            },
            py::return_value_policy::reference);

    py::implicitly_convertible<QGate, QProg>();
    py::implicitly_convertible<QCircuit, QProg>();
    py::implicitly_convertible<QIfProg, QProg>();
    py::implicitly_convertible<QWhileProg, QProg>();
    py::implicitly_convertible<QMeasure, QProg>();
    py::implicitly_convertible<QReset, QProg>();
    py::implicitly_convertible<ClassicalCondition, QProg>();

    /* hide */
    py::class_<HadamardQCircuit, QCircuit>(m, "hadamard_circuit")
        .def(py::init<QVec &>());

    py::class_<Encode>(m, "Encode")
        .def(py::init<>())
        .def("amplitude_encode",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::amplitude_encode),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("amplitude_encode",
             py::overload_cast<const QVec &, const std::vector<complex<double>> &>(&Encode::amplitude_encode),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("amplitude_encode_recursive",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::amplitude_encode_recursive),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("amplitude_encode_recursive",
             py::overload_cast<const QVec &, const QStat &>(&Encode::amplitude_encode_recursive),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("angle_encode",
             &Encode::angle_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg_v("gate_type", GateType::RY_GATE, "GateType.RY_GATE"),
             py::return_value_policy::automatic)
        .def("dense_angle_encode",
             &Encode::dense_angle_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("iqp_encode",
             &Encode::iqp_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg("control_list") = std::vector<std::pair<int, int>>{},
             py::arg("bool_inverse") = false,
             py::arg("repeats") = 1,
             py::return_value_policy::automatic)
        .def("basic_encode",
             &Encode::basic_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("dc_amplitude_encode",
             &Encode::dc_amplitude_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("schmidt_encode",
             &Encode::schmidt_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("bid_amplitude_encode",
             &Encode::bid_amplitude_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg("split") = 0,
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::map<std::string, std::complex<double>> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::vector<std::complex<double>> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::map<std::string, complex<double>> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::vector<complex<double>> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::map<std::string, complex<double>> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::vector<complex<double>> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data"),
             py::return_value_policy::automatic)
        .def("get_circuit", &Encode::get_circuit)
        .def("get_out_qubits", &Encode::get_out_qubits)
        .def("get_normalization_constant", &Encode::get_normalization_constant);

    /* use std::unique_ptr with py::nodelete as python object holder guarantee python object not own the C++ object */
    py::class_<OriginQubitPool, std::unique_ptr<OriginQubitPool, py::nodelete>>(m, "OriginQubitPool")
        .def(py::init(
            []()
            {
                return std::unique_ptr<OriginQubitPool, py::nodelete>(OriginQubitPool::get_instance());
            }))
        .def("get_capacity", &OriginQubitPool::get_capacity)
        .def("set_capacity", &OriginQubitPool::set_capacity)
        .def("get_qubit_by_addr",
             &OriginQubitPool::get_qubit_by_addr,
             py::arg("qubit_addr"),
             /* py::return_value_policy::reference_internal will guarantee the class life time longer than returned reference */
             py::return_value_policy::reference_internal)
        .def("clearAll", &OriginQubitPool::clearAll)
        .def("getMaxQubit", &OriginQubitPool::getMaxQubit)
        .def("getIdleQubit", &OriginQubitPool::getIdleQubit)
        .def("get_max_usedqubit_addr", &OriginQubitPool::get_max_usedqubit_addr)
        .def("allocateQubitThroughPhyAddress",
             &OriginQubitPool::allocateQubitThroughPhyAddress,
             py::arg("qubit_addr"),
             py::return_value_policy::reference_internal)
        .def("allocateQubitThroughVirAddress",
             &OriginQubitPool::allocateQubitThroughVirAddress,
             py::arg("qubit_num"),
             py::return_value_policy::reference_internal)
        .def("getPhysicalQubitAddr",
             &OriginQubitPool::getPhysicalQubitAddr,
             py::arg("qubit"))
        .def("getVirtualQubitAddress",
             &OriginQubitPool::getVirtualQubitAddress,
             py::arg("qubit"))
        .def(
            "get_allocate_qubits",
            [](OriginQubitPool &self)
            {
                QVec qubits;
                self.get_allocate_qubits(qubits);
                return qubits;
            },
            "get allocate qubits",
            py::return_value_policy::reference_internal)

        .def("qAlloc", &OriginQubitPool::qAlloc, py::return_value_policy::reference_internal)
        .def("qAlloc_many",
             &OriginQubitPool::qAllocMany,
             py::arg("qubit_num"),
             "Allocate a list of qubits",
             py::return_value_policy::reference_internal)
        .def("qFree", &OriginQubitPool::qFree)
        .def("qFree_all", py::overload_cast<QVec &>(&OriginQubitPool::qFreeAll))
        .def("qFree_all", py::overload_cast<>(&OriginQubitPool::qFreeAll));

    py::class_<OriginCMem, std::unique_ptr<OriginCMem, py::nodelete>>(m, "OriginCMem")
        .def(py::init(
            []()
            {
                return std::unique_ptr<OriginCMem, py::nodelete>(OriginCMem::get_instance());
            }))
        .def("get_capacity", &OriginCMem::get_capacity)
        .def("set_capacity", &OriginCMem::set_capacity)
        .def("get_cbit_by_addr",
             &OriginCMem::get_cbit_by_addr,
             py::arg("cbit_addr"),
             py::return_value_policy::reference_internal)
        .def("Allocate_CBit", py::overload_cast<>(&OriginCMem::Allocate_CBit), py::return_value_policy::reference_internal)
        .def("Allocate_CBit",
             py::overload_cast<size_t>(&OriginCMem::Allocate_CBit),
             py::arg("cbit_num"),
             py::return_value_policy::reference_internal)
        .def("getMaxMem", &OriginCMem::getMaxMem)
        .def("getIdleMem", &OriginCMem::getIdleMem)
        .def("Free_CBit",
             &OriginCMem::Free_CBit,
             py::arg("cbit"))
        .def("clearAll", &OriginCMem::clearAll)

        .def(
            "get_allocate_cbits",
            [](OriginCMem &self)
            {
                vector<ClassicalCondition> cc_vec;
                self.get_allocate_cbits(cc_vec);
                return cc_vec;
            },
            "Get allocate cbits",
            py::return_value_policy::reference_internal)

        .def("cAlloc", py::overload_cast<>(&OriginCMem::cAlloc))
        .def("cAlloc",
             py::overload_cast<size_t>(&OriginCMem::cAlloc),
             py::arg("cbit_num"))
        .def("cAlloc_many",
             &OriginCMem::cAllocMany,
             py::arg("count"))
        .def("cFree",
             &OriginCMem::cFree,
             py::arg("classical_cond"))
        .def("cFree_all",
             py::overload_cast<std::vector<ClassicalCondition> &>(&OriginCMem::cFreeAll),
             py::arg("classical_cond_list"))
        .def("cFree_all", py::overload_cast<>(&OriginCMem::cFreeAll));

    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();

    /*******************************************************************
     *                           QProgDAG
     */
    py::class_<QProgDAGEdge>(m, "QProgDAGEdge")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("from"),
             py::arg("to"),
             py::arg("qubit"))
        .def_readwrite("m_from", &QProgDAGEdge::m_from)
        .def_readwrite("m_to", &QProgDAGEdge::m_to)
        .def_readwrite("m_qubit", &QProgDAGEdge::m_qubit);

    py::class_<QProgDAGVertex>(m, "QProgDAGVertex")
        .def(py::init<>())
        .def_readwrite("m_id", &QProgDAGVertex::m_id)
        .def_readwrite("m_type", &QProgDAGVertex::m_type)
        .def_readwrite("m_layer", &QProgDAGVertex::m_layer)
        .def_readwrite("m_pre_node", &QProgDAGVertex::m_pre_node)
        .def_readwrite("m_succ_node", &QProgDAGVertex::m_succ_node)
        .def(
            "is_dagger",
            [](QProgDAGVertex &self)
            {
                return self.m_node->m_dagger;
            },
            py::return_value_policy::reference)
        .def(
            "get_iter",
            [](QProgDAGVertex &self)
            {
                return self.m_node->m_itr;
            },
            py::return_value_policy::reference_internal)
        .def(
            "get_qubits_vec",
            [](QProgDAGVertex &self)
            {
                return self.m_node->m_qubits_vec;
            },
            py::return_value_policy::reference_internal)
        .def(
            "get_control_vec",
            [](QProgDAGVertex &self)
            {
                return self.m_node->m_control_vec;
            },
            py::return_value_policy::reference_internal);

    py::class_<QProgDAG>(m, "QProgDAG")
        .def(py::init<>())
        .def("get_vertex_set",
             py::overload_cast<>(&QProgDAG::get_vertex),
             /* as vector is copied, but element of containor is reference */
             py::return_value_policy::reference_internal)
        .def("get_target_vertex",
             py::overload_cast<const size_t>(&QProgDAG::get_vertex, py::const_),
             py::arg("vertice_num"),
             py::return_value_policy::reference_internal)
        .def(
            "get_edges",
            [](QProgDAG &self)
            {
                auto edges_set = self.get_edges();
                std::vector<QProgDAGEdge> edges_vec;
                for (const auto &_e : edges_set)
                {
                    edges_vec.emplace_back(_e);
                }
                return edges_vec;
            },
            py::return_value_policy::automatic);

    py::class_<QuantumStateTomography>(m, "QuantumStateTomography")
        .def(py::init<>())
        .def("combine_qprogs",
             py::overload_cast<const QProg &, const QVec &>(&QuantumStateTomography::combine_qprogs<QProg>),
             py::arg("circuit"),
             py::arg("qlist"),
             "Return a list of quantum state tomography quantum programs.",
             py::return_value_policy::reference_internal)
        .def("combine_qprogs",
             py::overload_cast<const QCircuit &, const QVec &>(&QuantumStateTomography::combine_qprogs<QCircuit>),
             py::arg("circuit"), py::arg("qlist"),
             "Return a list of quantum state tomography quantum programs.",
             py::return_value_policy::reference_internal)
        .def("combine_qprogs",
             py::overload_cast<const QProg &, const std::vector<size_t> &>(&QuantumStateTomography::combine_qprogs<QProg>),
             py::arg("circuit"),
             py::arg("qlist"),
             "Return a list of quantum state tomography quantum programs.",
             py::return_value_policy::reference_internal)
        .def("combine_qprogs",
             py::overload_cast<const QCircuit &, const std::vector<size_t> &>(&QuantumStateTomography::combine_qprogs<QCircuit>),
             py::arg("circuit"),
             py::arg("qlist"),
             "Return a list of quantum state tomography quantum programs.",
             py::return_value_policy::reference_internal)
        .def("exec",
             &QuantumStateTomography::exec,
             py::arg("qm"),
             py::arg("shots"),
             "run state tomography QProgs",
             py::return_value_policy::reference_internal)
        .def("set_qprog_results",
             py::overload_cast<size_t, const std::vector<std::map<std::string, double>> &>(&QuantumStateTomography::set_qprog_results),
             py::arg("qlist"),
             py::arg("results"),
             "set combine_qprogs result",
             py::return_value_policy::reference_internal)
        .def("caculate_tomography_density",
             &QuantumStateTomography::caculate_tomography_density,
             "caculate tomography density",
             py::return_value_policy::reference_internal);

    py::class_<Fusion>(m, "Fusion")
        .def(py::init<>())
        .def("aggregate_operations",
             py::overload_cast<QCircuit &, QuantumMachine *>(&Fusion::aggregate_operations),
             py::arg("circuit"),
             py::arg("qvm"),
             py::return_value_policy::automatic)
        .def("aggregate_operations",
             py::overload_cast<QProg &, QuantumMachine *>(&Fusion::aggregate_operations),
             py::arg("circuit"),
             py::arg("qvm"),
             py::return_value_policy::automatic);

    py::class_<LatexMatrix>(m, "LatexMatrix",
                            "Generate quantum circuits latex src code can be compiled on latex package 'qcircuit'\n"
                            "circuits element treated as matrix element in latex syntax\n"
                            "\n"
                            "qcircuit package tutorial [https://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf]")
        .def(py::init<>())
        .def("set_label",
             &LatexMatrix::setLabel,
             py::arg("qubit_label"),
             py::arg("cbit_label") = LatexMatrix::Label(),
             py::arg("time_seq_label") = "",
             py::arg("head") = true,
             "Set Label at left most head col or right most tail col\n"
             "label can be reseted at any time\n"
             "\n"
             "Args:\n"
             "    qubit_label: label for qwire left most head label, at row, in latex syntax. not given row will keep empty\n"
             "                 eg. {0: 'q_{1}', 2:'q_{2}'}\n"
             "    cbit_label: classic label string, support latex str\n"
             "    time_seq_label: if given, set time squence lable\n"
             "    head: if true, label append head; false, append at tail")
        .def("set_logo",
             &LatexMatrix::setLogo,
             py::arg("logo") = "",
             "Add a logo string")
        .def("insert_gate",
             &LatexMatrix::insertGate,
             py::arg("target_rows"),
             py::arg("ctrl_rows"),
             py::arg("from_col"),
             py::arg("gate_type"),
             py::arg("gate_name") = "",
             py::arg("dagger") = false,
             py::arg("param") = "",
             "Insert a gate into circute\n"
             "\n"
             "Args:\n"
             "    target_rows: gate targets row of latex matrix\n"
             "    ctrl_rows"
             "    from_col: gate wanted col pos, but there may be not enough zone to put gate\n"
             "    gate_type: enum type of LATEX_GATE_TYPE\n"
             "    gate_name\n"
             "    dagger: dagger flag\n"
             "    param: gate param str\n"
             "\n"
             "Returns:\n"
             "    int: real col num. if there is no enough zone to put gate at 'from_col', we will find suitable col to put gate after 'from_col'")
        .def("insert_barrier",
             &LatexMatrix::insertBarrier,
             py::arg("rows"),
             py::arg("from_col"),
             "\n"
             "Args:\n"
             "    rows: rows need be barriered, may not continus")
        .def("insert_measure",
             &LatexMatrix::insertMeasure,
             py::arg("q_row"),
             py::arg("c_row"),
             py::arg("from_col"))
        .def("insert_reset",
             &LatexMatrix::insertReset,
             py::arg("q_row"),
             py::arg("from_col"))
        .def("insert_timeseq",
             &LatexMatrix::insertTimeSeq,
             py::arg("t_col"),
             py::arg("time_seq"),
             "Beware we do not check col num, may cause overwrite. user must take care col num self")
        .def("str",
             &LatexMatrix::str,
             py::arg("with_time") = false,
             "return final latex source code, can be called at any time")
        .def("__str__", &LatexMatrix::str);

    return;
}
