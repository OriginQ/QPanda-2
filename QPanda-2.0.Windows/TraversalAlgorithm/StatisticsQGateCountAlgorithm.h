#ifndef _STATISTICS_QGATE_COUNT_ALGORITHM
#define _STATISTICS_QGATE_COUNT_ALGORITHM
#include "QuantumCircuit/QProgram.h"
class StatisticsQGateCountAlgorithm
{
public:
    StatisticsQGateCountAlgorithm();
    ~StatisticsQGateCountAlgorithm();
    static size_t countQGate(AbstractQuantumCircuit *);
    
};

#endif // !_STATISTICS_QGATE_COUNT_ALGORITHM



