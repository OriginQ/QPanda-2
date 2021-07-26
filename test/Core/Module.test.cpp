#ifdef ABANDON
#include <map>
#include "QPanda.h"
#include "Core/Module/Operators.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>  
#include "gtest/gtest.h"
using namespace std;
USING_QPANDA
QCircuit test()
{
	QProg prog;
	qubits<10> qs;
	return h(qs);
}
TEST(Module, module_test1)
{
	std::cout << "======================================" << std::endl;
	/*auto qm = initQuantumMachine();
	Configuration config = { 10000,10000 };
	qm->setConfig(config);
	ModuleContext::setContext(qm);
	QProg prog;
	qubit q;
	prog << H(q);
	qubits<20> qs;
	prog << h(qs);

	qubits<20> qs1;
	qubits<20> qs2;
	prog << h(qs1);
	prog << h(qs2);
	prog << test();

	std::string s = transformQProgToOriginIR(prog, qm);
	cout << s<<endl;*/

	std::cout << "======================================" << std::endl;

	/*qvec qvec1(qs1);
	qvec qvec2(qs1);
	prog << h(qvec1);
	std::string s2 = transformQProgToOriginIR(prog, qm);
	cout << s2 << endl;
	std::cout << "qvec1+qvec2 size "<<(qvec1+qvec2).size() << std::endl;
	std::cout << "qvec1-qvec2 size " << (qvec1 - qvec2).size() << std::endl;*/
}


#endif //ABANDON
