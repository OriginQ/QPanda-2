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

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                                 .def(py::self OP cbit_size_t())\
                                                 .def(cbit_size_t() OP py::self)

void init_core_class(py::module & m)
{
    py::class_<NodeInfo>(m, "NodeInfo")
            .def(py::init<>())
		    .def(py::init<>([](const NodeIter iter, QVec target_qubits, QVec control_qubits,
			int type, const bool dagger) {
		        return NodeInfo(iter, target_qubits, control_qubits, type, dagger); }))
            .def_readwrite("m_iter", &NodeInfo::m_iter)
            .def_readwrite("m_node_type", &NodeInfo::m_node_type)
            .def_readwrite("m_gate_type", &NodeInfo::m_gate_type)
            .def_readwrite("m_is_dagger", &NodeInfo::m_is_dagger)
            .def_readwrite("m_target_qubits", &NodeInfo::m_target_qubits)
            .def_readwrite("m_control_qubits", &NodeInfo::m_control_qubits)
		    .def_readwrite("m_cbits", &NodeInfo::m_cbits)
		    .def_readwrite("m_params", &NodeInfo::m_params)
		    .def_readwrite("m_name", &NodeInfo::m_name)
            .def("reset", &NodeInfo::reset);

    py::enum_<SingleGateTransferType>(m, "SingleGateTransferType")
            .value("SINGLE_GATE_INVALID", SINGLE_GATE_INVALID)
            .value("ARBITRARY_ROTATION", ARBITRARY_ROTATION)
            .value("DOUBLE_CONTINUOUS", DOUBLE_CONTINUOUS)
            .value("SINGLE_CONTINUOUS_DISCRETE", SINGLE_CONTINUOUS_DISCRETE)
            .value("DOUBLE_DISCRETE", DOUBLE_DISCRETE)
            .export_values();

    py::enum_<DoubleGateTransferType>(m, "DoubleGateTransferType")
            .value("DOUBLE_GATE_INVALID", DOUBLE_GATE_INVALID)
            .value("DOUBLE_BIT_GATE", DOUBLE_BIT_GATE)
            .export_values();

    py::class_<Qubit>(m, "Qubit")
        .def("getPhysicalQubitPtr", &Qubit::getPhysicalQubitPtr, py::return_value_policy::reference)
        .def("get_phy_addr", &Qubit::get_phy_addr, py::return_value_policy::reference)
        ;

    py::class_<PhysicalQubit>(m, "PhysicalQubit")
        .def("getQubitAddr", &PhysicalQubit::getQubitAddr, py::return_value_policy::reference)
        ;

    py::class_<CBit>(m, "CBit")
        .def("getName", &CBit::getName);

    py::class_<ClassicalCondition>(m, "ClassicalCondition")
        .def("get_val", &ClassicalCondition::get_val,"get value")
        .def("set_val",&ClassicalCondition::set_val,"set value")
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<=)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>=)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(+)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(-)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(*)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(/ )
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(== )
        ;

    py::enum_<QError>(m, "QError")
        .value("UndefineError", QError::undefineError)
        .value("qErrorNone", QError::qErrorNone)
        .value("qParameterError", QError::qParameterError)
        .value("qubitError", QError::qubitError)
        .value("loadFileError", QError::loadFileError)
        .value("initStateError", QError::initStateError)
        .value("destroyStateError", QError::destroyStateError)
        .value("setComputeUnitError", QError::setComputeUnitError)
        .value("runProgramError", QError::runProgramError)
        .value("getResultError", QError::getResultError)
        .value("getQStateError", QError::getQStateError)
        ;

    py::class_<OriginCollection>(m, "OriginCollection")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<OriginCollection>())
        .def("setNames", [](OriginCollection& c, py::args args) {
            std::vector<std::string> all_key;
            for (auto arg : args) { all_key.push_back(std::string(py::str(arg))); }
            c = all_key;
        })
        .def("insertValue", [](OriginCollection& c, std::string key, py::args args) {
            int i = 1;
            auto vector = c.getKeyVector();
            c.addValue(vector[0], key);
            for (auto arg : args) {
                c.addValue(vector[i],std::string(py::str (arg)));
                i++;
                }
        })
        .def("insertValue", [](OriginCollection& c, int key, py::args args) {
        int i = 1;
        auto vector = c.getKeyVector();
        c.addValue(vector[0], key);
        for (auto arg : args) {
            c.addValue(vector[i], std::string(py::str(arg)));
            i++;
        }
    })
        .def("getValue",&OriginCollection::getValue,"Get value by Key name")
        .def("getValueByKey", [](OriginCollection & c, std::string key_value) {
        return c.getValueByKey(key_value);
    }, "Get Value by key value")
        .def("getValueByKey", [](OriginCollection & c, int key_value) {
        return c.getValueByKey(key_value);
    }, "Get Value by key value")
        .def("open", &OriginCollection::open, "Open json file")
        .def("write", &OriginCollection::write, "write json file")
        .def("getJsonString", &OriginCollection::getJsonString, "Get Json String")
        .def("getFilePath", &OriginCollection::getFilePath, "Get file path")
        .def("getKeyVector", &OriginCollection::getKeyVector, "Get key vector");

    py::class_<QResult>(m, "QResult")
        .def("getResultMap", &QResult::getResultMap, py::return_value_policy::reference);

    py::class_<ClassicalProg>(m, "ClassicalProg")
            .def(py::init<ClassicalCondition &>());

    py::class_<QProg>(m, "QProg")
            .def(py::init<>())
            .def(py::init<QProg&>())
            .def(py::init<QCircuit &>())
            .def(py::init<QIfProg &>())
            .def(py::init<QWhileProg &>())
            .def(py::init<QGate &>())
            .def(py::init<QMeasure &>())
            .def(py::init<QReset &>())
            .def(py::init<ClassicalCondition &>())
            .def(py::init([](NodeIter & iter) {
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
                 }
                 ))
           
            .def("__lshift__", &QProg::operator<<<QProg >,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QGate>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QCircuit>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QIfProg>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QWhileProg>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QMeasure>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<QReset>,
                py::return_value_policy::reference)
            .def("__lshift__", &QProg::operator<<<ClassicalCondition>,
                py::return_value_policy::reference)

            .def("is_empty", &QProg::is_empty,
                 py::return_value_policy::automatic_reference)

            .def("insert", &QProg::operator<<<QProg >,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QGate>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QCircuit>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QIfProg>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QWhileProg>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QMeasure>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<QReset>,
                 py::return_value_policy::reference)
            .def("insert", &QProg::operator<<<ClassicalCondition>,
                 py::return_value_policy::reference)
            .def("begin",&QProg::getFirstNodeIter,
                 py::return_value_policy::reference)
            .def("end",&QProg::getEndNodeIter,
                 py::return_value_policy::reference)
            .def("last",&QProg::getLastNodeIter,
                 py::return_value_policy::reference)
					 .def("__str__", [](QProg& p) {
					 auto text_pic_str = draw_qprog(p);
#if defined(WIN32) || defined(_WIN32)
					 text_pic_str = fit_to_gbk(text_pic_str);
					 //text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
					 return  text_pic_str; },
    py::return_value_policy::reference)
            .def("head",&QProg::getHeadNodeIter,
                 py::return_value_policy::reference);


    py::implicitly_convertible<QGate, QProg>();
    py::implicitly_convertible<QCircuit, QProg>();
    py::implicitly_convertible<QIfProg, QProg>();
    py::implicitly_convertible<QWhileProg, QProg>();
    py::implicitly_convertible<QMeasure, QProg>();
    py::implicitly_convertible<QReset, QProg>();
    py::implicitly_convertible<ClassicalCondition, QProg>();

    py::class_<QCircuit>(m, "QCircuit")
            .def(py::init<>())
            .def(py::init([](NodeIter & iter) {
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
                 }
                 )) 

            .def("__lshift__", &QCircuit::operator<< <QCircuit>,
                py::return_value_policy::reference)
            .def("__lshift__", &QCircuit::operator<< <QGate>,
                py::return_value_policy::reference)

            .def("is_empty", &QCircuit::is_empty,
                 py::return_value_policy::automatic)

            .def("insert", &QCircuit::operator<< <QCircuit>,
                 py::return_value_policy::reference)
            .def("insert", &QCircuit::operator<< <QGate>,
                 py::return_value_policy::reference)
            .def("dagger", &QCircuit::dagger,
                 py::return_value_policy::automatic)
            .def("control", &QCircuit::control,
                 py::return_value_policy::automatic)
            .def("set_dagger", &QCircuit::setDagger)
            .def("set_control", &QCircuit::setControl)
            .def("begin",&QCircuit::getFirstNodeIter,
                 py::return_value_policy::reference)
            .def("end",&QCircuit::getEndNodeIter,
                 py::return_value_policy::reference)
            .def("last",&QCircuit::getLastNodeIter,
                 py::return_value_policy::reference)
            .def("head",&QCircuit::getHeadNodeIter,
                 py::return_value_policy::reference)
					 .def("__str__", [](QCircuit& p) {
					 auto text_pic_str = draw_qprog(p);
#if defined(WIN32) || defined(_WIN32)
					 text_pic_str = fit_to_gbk(text_pic_str);
					 //text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
					 return  text_pic_str; },
                 py::return_value_policy::reference);

    /* hide */
    py::class_<HadamardQCircuit, QCircuit>(m, "hadamard_circuit")
            .def(py::init<QVec&>());

    py::class_<QGate>(m, "QGate")
            .def(py::init([](NodeIter & iter) {
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
                 }
                 ))
            .def("dagger", &QGate::dagger)
            .def("control", &QGate::control)
            .def("is_dagger", &QGate::isDagger)
            .def("set_dagger", &QGate::setDagger)
            .def("set_control", &QGate::setControl)
            .def("get_target_qubit_num", &QGate::getTargetQubitNum)
            .def("get_control_qubit_num", &QGate::getControlQubitNum)
            .def("get_qubits",&QGate::getQuBitVector,py::return_value_policy::automatic)
            .def("get_control_qubits", &QGate::getControlVector,py::return_value_policy::automatic)
            .def("gate_type", [](QGate & qgate) {
        return qgate.getQGate()->getGateType();
    })
    .def("gate_matrix", [](QGate & qgate) {
        QStat matrix;
        qgate.getQGate()->getMatrix(matrix);
        return matrix;
    }, py::return_value_policy::automatic);


    py::class_<QIfProg>(m, "QIfProg")
            .def(py::init([](NodeIter & iter) {
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
                 }
                 ))
            .def(py::init<ClassicalCondition &, QProg>())
            .def(py::init<ClassicalCondition &, QProg, QProg>())
            .def("get_true_branch", [](QIfProg & self) {
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
            .def("get_false_branch", [](QIfProg & self) {
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
            .def("get_classical_condition", &QIfProg::getClassicalCondition,
                 py::return_value_policy::automatic);


    py::class_<QWhileProg>(m, "QWhileProg")
            .def(py::init([](NodeIter & iter) {
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
                 }
                 ))
            .def(py::init<ClassicalCondition, QProg>())
            .def("get_true_branch", [](QWhileProg & self) {
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
            .def("get_classical_condition", &QWhileProg::getClassicalCondition,
                 py::return_value_policy::automatic);;

    py::class_<QMeasure>(m, "QMeasure")
            .def(py::init([](NodeIter & iter) {
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
            .def(py::init([](NodeIter & iter) {
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

    py::enum_<NodeType>(m, "NodeType")
        .value("NODE_UNDEFINED", NodeType::NODE_UNDEFINED)
        .value("GATE_NODE", NodeType::GATE_NODE)
        .value("CIRCUIT_NODE", NodeType::CIRCUIT_NODE)
        .value("PROG_NODE", NodeType::PROG_NODE)
        .value("MEASURE_GATE", NodeType::MEASURE_GATE)
        .value("WHILE_START_NODE", NodeType::WHILE_START_NODE)
        .value("QIF_START_NODE", NodeType::QIF_START_NODE)
        .value("CLASS_COND_NODE", NodeType::CLASS_COND_NODE)
        .value("RESET_NODE", NodeType::RESET_NODE);


    py::class_<NodeIter>(m, "NodeIter")
        .def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("get_next", &NodeIter::getNextIter, py::return_value_policy::automatic)
        .def("get_pre", [](NodeIter & iter) {
        return --iter;
        }, py::return_value_policy::reference)
        .def("get_node_type", [](NodeIter & iter) {
            auto node_type = (*iter)->getNodeType();
            return node_type;
        },
                py::return_value_policy::automatic);

    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();


	py::enum_<QCircuitOPtimizerMode>(m, "QCircuitOPtimizerMode")
		.value("Merge_H_X", QCircuitOPtimizerMode::Merge_H_X)
		.value("Merge_U3", QCircuitOPtimizerMode::Merge_U3)
		.value("Merge_RX", QCircuitOPtimizerMode::Merge_RX)
		.value("Merge_RY", QCircuitOPtimizerMode::Merge_RY)
		.value("Merge_RZ", QCircuitOPtimizerMode::Merge_RZ)
		.def("__or__", [](QCircuitOPtimizerMode& self, QCircuitOPtimizerMode& other) {return self | other; },
			py::return_value_policy::reference);

	py::enum_<DecompositionMode>(m, "DecompositionMode")
		.value("QR", DecompositionMode::QR)
		.value("HOUSEHOLDER_QR", DecompositionMode::HOUSEHOLDER_QR);

    return ;
}
