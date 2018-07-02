#ifndef _QRIGIN_QUANTUM_MACHINE_H
#define _QRIGIN_QUANTUM_MACHINE_H
#include "QuantumMachineInterface.h"
#include <vector>
using std::vector;
using std::string;
class OriginQMachine //:public QuantumMachine
{
public:
    OriginQMachine();
    ~OriginQMachine();

    void init(); // to initialize the quantum machine
    Qubit* Allocate_Qubit(); // allocate and return a qubit
    CBit* Allocate_CBit(); // allocate and run a cbit
    void Free_Qubit(Qubit*); // free a qubit
    void Free_CBit(CBit*); // free a cbit
    void load(QProg &); // load a qprog
    void append(QProg&); // append the qprog after the original
    void run(); // run on the quantum machine
    QMachineStatus* getStatus() const; // get the status of the quantum machine
    QResult* getResult(); // get the result of the quantum program
    void finalize(); // finish the program
private:
    QubitPool * m_pQubitPool = nullptr;
    CMem * m_pCMem = nullptr;
    int m_iQProgram = -1;;
    QResult* m_pQResult = nullptr;
    QMachineStatus* m_pQMachineStatus = nullptr;

    struct Configuration
    {
        size_t maxQubit;
        size_t maxCMem ;
    };
    Configuration m_Config;
    vector<vector<int>> m_qubitMatrix;
    vector<string> m_sSingleGateVector;
    vector<string> m_sDoubleGateVector;
};

#endif

