#ifndef _QUBIT_REFERENCE_H
#define _QUBIT_REFERENCE_H
#include <vector>
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN
class QubitReference :public Qubit
{
private:
    std::shared_ptr<CExpr> m_cepr;
    std::vector<Qubit *> m_qvec;
public:
    inline PhysicalQubit* getPhysicalQubitPtr() 
    {
        auto temp = m_cepr->eval();
        return m_qvec[temp]->getPhysicalQubitPtr();
    }

    inline QubitReference(ClassicalCondition & cc,std::vector<Qubit *> qvec)
    {
        m_cepr = cc.getExprPtr();
        
        for(auto aiter : qvec)
        {
            m_qvec.push_back(aiter);
        }
    }

    inline bool getOccupancy()
    {
        return true;
    }
    
    inline QubitReference(const QubitReference & old)
    {
        m_cepr = old.m_cepr;
        m_qvec.clear();
        for(auto aiter : m_qvec)
        {
            m_qvec.push_back(aiter);
        }
    }



    ~QubitReference()
    {
        m_cepr.reset();
    }

};
QPANDA_END
#endif 
