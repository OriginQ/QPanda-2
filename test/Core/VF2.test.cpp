#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <regex>
#include <ctime>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"


#include "Extensions/Extensions.h"
#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA


const std::string json = R"(
{
  "pattern": {
    "test": [
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "CNOT": [ 0, 1 ]
          }
        },
        "dst": {
          "cost": 0,
          "circuit": {

          }
        }
      },
      {
        "qubits": 3,
        "src": {
          "cost": 6,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "CNOT": [ 1, 2 ],
            "CNOT": [ 0, 1 ],
            "CNOT": [ 1, 2 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 2 ]
          }
        },
        "location": "default"
      },
      {
        "qubits": 2,
        "src": {
          "cost": 4,
          "circuit": {
            "X": [ 1 ],
            "CNOT": [ 0, 1 ],
            "X": [ 1 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 1 ]
          }
        },
        "location": "default"
      },
      {
        "qubits": 2,
        "src": {
          "cost": 6,
          "circuit": {
            "CNOT": [ 0, 2 ],
            "CNOT": [ 0, 1 ],
            "CNOT": [ 0, 2 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 1 ]
          }
        },
        "location": "default"
      }
    ],
    "nopara": [
    {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "S": [ 0 ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U1": [0, "PI/2"]
          }
        }
      },
      {
        "qubits": 3,
        "src": {
          "cost": 6,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "CNOT": [ 1, 2 ],
            "CNOT": [ 0, 1 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 2 ]
          }
        },
        "location": "default"
      },
      {
        "qubits": 2,
        "src": {
          "cost": 4,
          "circuit": {
            "X": [ 1 ],
            "CNOT": [ 0, 1 ],
            "X": [ 1 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 1 ]
          }
        },
        "location": "default"
      },
      {
        "qubits": 2,
        "src": {
          "cost": 10,
          "circuit": {
            "RZ": [ 1, "theta_1" ],
            "CNOT": [ 0, 1 ],
            "RZ": [ 1, "theta_2" ],
            "CNOT": [ 0, 1 ],
            "RZ": [ 0, "PI/4" ],
            "RZ": [ 1, "theta_3" ],
            "CNOT": [ 1, 0 ]
          }
        },
        "dst": {
          "cost": 9,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "RZ": [ 1, "theta_2" ],
            "CNOT": [ 0, 1 ],
            "RZ": [ 0, "PI/4" ],
            "RZ": [ 1, "theta_1+theta_3" ],
            "CNOT": [ 1, 0 ]
          }
        },
        "location": "default"
      },
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "RZ": [ 0, "theta_1" ],
            "RZ": [ 0, "theta_2" ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "RZ": [ 0, "theta_1+theta_2" ]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "CNOT": [ 0, 1 ]
          }
        },
        "dst": {
          "cost": 0,
          "circuit": {

          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "H": [ 0 ],
            "H": [ 0 ]
          }
        },
        "dst": {
          "cost": 0,
          "circuit": {

          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "X": [ 0 ],
            "H": [ 0 ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U2" : [ 0, "-PI", "-PI" ]
          }
        }
      },
       { 
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "H": [ 0 ],
            "X": [ 0 ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U2" : [ 0, "0", "0" ]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "H": [ 0 ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
			"U2":[0,"0", "PI"]
          }
        },
		"location": "default"
      },
      {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "RPhi": [ 0 ,"theta_1","theta_2"]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U3": [0, "theta_1", "theta_2 - PI/2", "-theta_2 + PI/2"]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "RY": [0, "theta_1"] 
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
			"RPhi": [ 0,"theta_1","PI/2" ]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 2,
          "circuit": {
            "U2": [ 0, "-PI", "-PI" ],
            "U2": [ 0, "0",  "0"  ]
          }
        },
        "dst": {
          "cost": 0,
          "circuit": {

          }
        }
      },
		{
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "RZ": [ 0, "theta_1" ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U1": [0, "theta_1" ]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "RX": [0, "theta_1"]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "RPhi": [ 0 ,"theta_1", "0"]
          }
        }
      },
      {
        "qubits": 1,
        "src": {
          "cost": 1,
          "circuit": {
            "U1": [0, "theta_1" ]
          }
        },
        "dst": {
          "cost": 1,
          "circuit": {
            "U3": [ 0, "theta_1","0","0" ]
          }
        }
      },
      {
        "qubits": 3,
        "src": {
          "cost": 6,
          "circuit": {
            "CNOT": [ 0, 1 ],
            "CNOT": [ 1, 2 ],
            "CNOT": [ 0, 1 ],
            "CNOT": [ 1, 2 ]
          }
        },
        "dst": {
          "cost": 2,
          "circuit": {
            "CNOT": [ 0, 2 ]
          }
        },
        "location": "default"
      }
    ]
  }
}
)";
const std::string qasm_string = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[10], q[11];
cx q[10], q[11];
cx q[10], q[11];
cx q[12], q[11];
cx q[15], q[12];
cx q[10], q[12];
cx q[10], q[12];
cx q[5], q[12];
cx q[18], q[5];
cx q[7], q[18];
cx q[16], q[18];
cx q[13], q[16];
cx q[17], q[13];
cx q[13], q[16];
cx q[13], q[16];
cx q[7], q[16];
cx q[7], q[16];
cx q[16], q[14];
cx q[0], q[16];
cx q[0], q[16];
cx q[7], q[0];
cx q[7], q[18];
cx q[7], q[18];
cx q[7], q[16];
cx q[7], q[18];
cx q[7], q[0];
cx q[7], q[0];
cx q[7], q[16];
cx q[7], q[18];
cx q[7], q[18];
cx q[16], q[18];
cx q[18], q[1];
cx q[18], q[4];
cx q[3], q[4];
cx q[16], q[3];
cx q[16], q[18];
cx q[18], q[4];
cx q[18], q[1];
cx q[7], q[18];
cx q[7], q[16];
cx q[7], q[18];
cx q[7], q[18];
cx q[18], q[1];
cx q[18], q[4];
cx q[1], q[4];)";


#if 1

void vertices_output(std::shared_ptr<QProgDAG> dag) {
	std::cout << "=========================================" << std::endl;
	std::cout << "Vertice:" << std::endl;
	for (auto i = 0; i < dag->get_vertex().size(); ++i) {
		auto vertice = dag->get_vertex(i);
		std::cout << "\tGate " << i << " GateType: " << vertice.m_type << ", the qubits are ";
		for (int i = 0; i < vertice.m_node->m_qubits_vec.size(); ++i) {
			std::cout << vertice.m_node->m_qubits_vec[i]->get_phy_addr() << " ";
		}
		std::cout << std::endl;
		std::cout << "The Parameter for angle(s) are ";
		for (auto& _angle : vertice.m_node->m_angles) {
			//std::cout << _angle << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "=========================================" << std::endl;
}

void edges_output(std::shared_ptr<QProgDAG> dag) {
	std::cout << "========================================" << std::endl;
	std::cout << "Edges:" << std::endl;
	for (auto i = 0; i < dag->get_vertex().size(); ++i) {
		auto vertice = dag->get_vertex(i);
		for (auto& _edge : vertice.m_pre_edges) {
			std::cout << _edge.m_from << "->" << _edge.m_to << ", qubit:" << _edge.m_qubit << std::endl;
		}
		for (auto& _edge : vertice.m_succ_edges) {
			std::cout << _edge.m_from << "->" << _edge.m_to << ", qubit:" << _edge.m_qubit << std::endl;
		}
	}
}

#endif
static bool test_vf2_1()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << CNOT(q[0], q[1]) << CNOT(q[0],q[1]) << H(q[0]);
	prog << circuit << MeasureAll(q, c);

	//std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, json, 1);
	//std::cout << "result_prog1: " << prog.get_qgate_num();
    if (prog.get_qgate_num() != 0)
        return false;
    else 
	    return true;
}

static bool test_vf2_3() {
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << RZ(q[2], PI / 4) << CNOT(q[0], q[2]) << RZ(q[1], PI / 4) << RZ(q[1], PI / 6) << 
		RZ(q[2], PI / 6) << CNOT(q[0], q[2]) << RZ(q[0], PI / 4) << RZ(q[2], PI / 4) << CNOT(q[2], q[0]);
	prog << circuit << MeasureAll(q, c);

	//std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, json, 1);
	//std::cout << "result_prog: " << prog;

	return 1;
	//std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, json, 1);
    if (prog.get_qgate_num() != 1)
        return false;
    else 
	    return true;
}

static bool test_vf2_1_1() {

	return 1;
} 


static bool test_vf2_1_2() {
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << X(q[0]) << CNOT(q[2], q[1]) << CNOT(q[2], q[1]) << H(q[0]) << CNOT(q[0], q[2]) << CNOT(q[0], q[2]) << CNOT(q[1], q[2]);
	prog << circuit;

	//std::cout << "src_prog: "<< prog;

	/*QCircuitRewrite rewriter;
	auto new_prog = rewriter.circuitRewrite(prog);*/
	sub_cir_replace(prog, json, 1);
	//std::cout << "result_prog555: " << prog.get_qgate_num();
    if (prog.get_qgate_num() != 0)
        return false;
    else
	    return true;
}

TEST(VF2, test1)
{
	bool test_val = false;
	try
	{
        test_val = test_vf2_1();
        test_val = test_vf2_3();
		test_val = test_vf2_1_2();
        
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	//ASSERT_TRUE(test_val);

	//cout << "VF2 test over, press Enter to continue." << endl;
	//getchar();
}

#endif
