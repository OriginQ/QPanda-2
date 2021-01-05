#include <string.h>
#include <iostream>
#include "Components/Operator/PauliOperator.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "QAlg/QAOA/QAOA.h"

USING_QPANDA

double myFunc(const std::string& key, const PauliOperator& pauli)
{
    double sum = 0;

    QHamiltonian hamiltonian = pauli.toHamiltonian();

    for_each(hamiltonian.begin(),
        hamiltonian.end(),
        [&](const QHamiltonianItem& item)
        {
            std::vector<size_t> index_vec;
            for (auto iter = item.first.begin();
                iter != item.first.end();
                iter++)
            {
                index_vec.push_back(iter->first);
            }

            //		double value = item.second;
            size_t i = index_vec.front();
            size_t j = index_vec.back();
            if (key[i] != key[j])
            {
                sum += item.second;
            }
        });

    return sum;
}

auto getHamiltonian()
{
    //return PauliOperator({
    //    {"Z0 Z4", 0.73},{"Z2 Z5", 0.88},
    //    {"Z0 Z5", 0.33},{"Z2 Z6", 0.58},
    //    {"Z0 Z6", 0.50},{"Z3 Z5", 0.67},
    //    {"Z1 Z4", 0.69},{"Z3 Z6", 0.43},
    //    {"Z1 Z5", 0.36}
    //});

    return PauliOperator(PauliOperator::PauliMap{ {
        {"Z0 Z6", 0.49},
        {"Z6 Z1", 0.59},
        {"Z1 Z7", 0.44},
        {"Z7 Z2", 0.56},
        {"Z2 Z8", 0.63},
        {"Z8 Z13", 0.36},
        {"Z13 Z19", 0.81},
        {"Z19 Z14", 0.29},
        {"Z14 Z9", 0.52},
        {"Z9 Z4", 0.43},
        {"Z13 Z18", 0.72},
        {"Z18 Z12", 0.40},
        {"Z12 Z7", 0.60},
        {"Z12 Z17", 0.71},
        {"Z17 Z11", 0.50},
        {"Z11 Z6", 0.64},
        {"Z11 Z16", 0.57},
        {"Z16 Z10", 0.41},
        {"Z10 Z5", 0.23},
        {"Z10 Z15", 0.40},
        {"Z5 Z0", 0.18}} });
}

int main()
{
    QAOA qaoa;
    auto hamiltonian = getHamiltonian();
    qaoa.setHamiltonian(hamiltonian);
    qaoa.setStep(3);
    qaoa.setShots(1000);
    qaoa.getOptimizer()->setDisp(true);
    qaoa.regiestUserDefinedFunc(std::bind(&myFunc,
        std::placeholders::_1,
        hamiltonian));

    return qaoa.exec();
}
