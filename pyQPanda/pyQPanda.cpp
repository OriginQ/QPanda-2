#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../QPanda-2.0.Core/QPanda.h"
#include <map>
using std::map;
namespace py = pybind11;

PYBIND11_MODULE(pyQPanda, m)
{
    m.doc() = "";
    m.def("init",
        &init,
        "to init the environment. Use this at the beginning"
	);

    m.def("finalize", &finalize,
        "to finalize the environment. Use this at the end"
    );

    m.def("qAlloc", []() {return qAlloc();},
        "Allocate a qubit",
		py::return_value_policy::reference
    );

    m.def("qAlloc", [](size_t size) {return qAlloc(size);}, 
		"Allocate several qubits",
        py::return_value_policy::reference
    );

    m.def("qFree", &qFree, "Free a qubit");

    m.def("cAlloc", []() {return cAlloc(); },
		"Allocate a CBit",
        py::return_value_policy::reference
    );

    m.def("cFree", &cFree, "Free a CBit");

    m.def("load", &load, "load a program");

    m.def("append", &append,
		"append a program after the loaded program");

    m.def("getstat", &getstat,
		"get the status(ptr) of the quantum machine");

    m.def("getResult", &getResult, 
		"get the result(ptr)");

    m.def("getAllocateQubitNum", &getAllocateQubitNum,
		"getAllocateQubitNum");

    m.def("getAllocateCMem", &getAllocateCMem, "getAllocateCMem");

    m.def("getCBitValue", &getCBitValue, "getCBitValue");

    m.def("bind_a_cbit", &bind_a_cbit, 
		"bind a cbit to a classicalcondition variable");

    m.def("run", &run, "run the loaded program");

    m.def("getResultMap", &getResultMap,
		"Directly get the result map");

    m.def("CreateEmptyQProg", &CreateEmptyQProg, 
		"Create an empty QProg Container",
        py::return_value_policy::automatic
    );


    m.def("CreateWhileProg", [](ClassicalCondition& m, QProg & qn)
				{QNode * node = (QNode *)&qn;
				 return CreateWhileProg(m, node);}, 
		"Create a WhileProg",
        py::return_value_policy::automatic
    );

    m.def("CreateWhileProg", [](ClassicalCondition& m, QCircuit & qn)
				{QNode * node = (QNode *)&qn; 
				 return CreateWhileProg(m, node);}, 
		"Create a WhileProg",
        py::return_value_policy::automatic
    );

    m.def("CreateWhileProg", [](ClassicalCondition& m, QGate & qn)
				{QNode * node = (QNode *)&qn; 
				 return CreateWhileProg(m, node);},
		"Create a WhileProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition& m, QProg & qn) 
				{QNode * node = (QNode *)&qn; 
				 return CreateIfProg(m, node);},
		"Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition& m, QCircuit & qn)
				{QNode * node = (QNode *)&qn; 
				 return CreateIfProg(m, node); }, 
		"Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition& m, QGate & qn) 
				{QNode * node = (QNode *)&qn; 
				 return CreateIfProg(m, node);}, 
		"Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QProg & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);}, 
		"Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QCircuit & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);}, 
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QGate & qn2)
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);},
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QGate & qn2)
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);},
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QCircuit & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);}, 
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QProg & qn2)
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);},
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QGate & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2; 
				 return CreateIfProg(m, node1, node2);}, 
		"Create a IfProg",
        py::return_value_policy::automatic
	);


    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QCircuit & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2;
				 return CreateIfProg(m, node1, node2);},
		"Create a IfProg",
        py::return_value_policy::automatic
	);
	
    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QProg & qn2) 
				{QNode * node1 = (QNode *)&qn1;
				 QNode * node2 = (QNode *)&qn2;
				 return CreateIfProg(m, node1, node2);},
		"Create a IfProg",
        py::return_value_policy::automatic
	);

    m.def("CreateEmptyCircuit", &CreateEmptyCircuit,
		"Create an empty QCircuit Container",
        py::return_value_policy::automatic
    );

    m.def("Measure", &Measure, 
		"Create a Measure operation",
        py::return_value_policy::automatic
    );

    m.def("H", &H, "Create a H gate",
        py::return_value_policy::automatic
    );

    m.def("T", &T, "Create a T gate",
        py::return_value_policy::automatic
    );

    m.def("S", &S, "Create a S gate",
        py::return_value_policy::automatic
    );

    m.def("X", &X, "Create an X gate",
        py::return_value_policy::automatic
    );

    m.def("Y", &Y, "Create a Y gate",
        py::return_value_policy::automatic
    );

    m.def("Z", &Z, "Create a Z gate",
        py::return_value_policy::automatic
    );

    m.def("X1", &X1, "Create an X1 gate",
        py::return_value_policy::automatic
    );

    m.def("Y1", &Y1, "Create a Y1 gate",
        py::return_value_policy::automatic
    );

    m.def("Z1", &Z1, "Create a Z1 gate",
        py::return_value_policy::automatic
    );

    m.def("RX", &RX, "Create a RX gate",
        py::return_value_policy::automatic
    );

    m.def("RY", &RY, "Create a RY gate",
        py::return_value_policy::automatic
    );

    m.def("RZ", &RZ, "Create a RZ gate",
        py::return_value_policy::automatic
    );

    m.def("CNOT", &CNOT, "Create a CNOT gate",
        py::return_value_policy::automatic
    );

    m.def("CZ", &CZ, "Create a CZ gate",
        py::return_value_policy::automatic
    );

    m.def("U4", [](QStat & matrix, Qubit *qubit)
			{return U4(matrix, qubit); }, 
		"Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("U4", [](double alpha, double beta, double gamma, double delta, 
					Qubit * qbit)
			{return U4(alpha, beta, gamma, delta, qbit); }, 
		"Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](double alpha, double beta, double gamma, double delta,
					Qubit * controlQBit, Qubit * targetQBit)
			{return CU(alpha, beta, gamma, delta, controlQBit, targetQBit);}, 
		"Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](QStat & matrix, Qubit * controlQBit, Qubit * targetQBit)
			{return CU(matrix, controlQBit, targetQBit); },
		"Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("iSWAP", 
		[](Qubit * controlQBit, Qubit * targetQBit)
			{return iSWAP(controlQBit, targetQBit); },
		"Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("iSWAP", 
		[](Qubit * controlQBit, Qubit * targetQBit, double theta)
			{return iSWAP(controlQBit, targetQBit, theta); }, 
		"Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("CR", &CR, "Create a CR gate",
        py::return_value_policy::automatic
    );

    m.def("qRunesProg", &qProgToQRunes, "QProg to QRunes",
        py::return_value_policy::automatic_reference
    );

	m.def("PMeasure", &PMeasure, 
		"Get the probability distribution over qubits",
		py::return_value_policy::automatic
	);

	m.def("PMeasure_no_index", &PMeasure_no_index,
		"Get the probability distribution over qubits",
		py::return_value_policy::automatic
	);

	m.def("accumulateProbability", &accumulateProbability,
		"Accumulate the probability from a prob list",
		py::return_value_policy::automatic
	);

	m.def("quick_measure", &quick_measure,
		"Use PMeasure instead of run the program repeatedly",
		py::return_value_policy::automatic
	);

   py::class_<QProg>(m, "QProg") 
       .def(py::init<>())
       .def("insert", &QProg::operator<<<QGate>) 
       .def("insert", &QProg::operator<<<QProg>) 
       .def("insert", &QProg::operator<<<QCircuit>) 
       .def("insert", &QProg::operator<<<QMeasure>) 
       .def("insert", &QProg::operator<<<QIfProg>) 
       .def("insert", &QProg::operator<<<QWhileProg>);

    py::class_<QCircuit>(m, "QCircuit")
        .def(py::init<>())
        .def("insert", &QCircuit::operator<< <QCircuit>)
        .def("insert", &QCircuit::operator<< <QGate>)
        .def("dagger", &QCircuit::dagger)
        .def("control",&QCircuit::control);

    py::class_<QGate>(m, "QGate")
        .def("dagger", &QGate::dagger)
        .def("control",&QGate::control);

	py::class_<QIfProg>(m, "QIfProg");
	py::class_<QWhileProg>(m, "QWhileProg");
    py::class_<QMeasure>(m, "QMeasure");
    py::class_<Qubit>(m, "Qubit");
    py::class_<CBit>(m, "CBit");
    py::class_<ClassicalCondition>(m, "ClassicalCondition");
}