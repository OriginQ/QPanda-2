#ifndef _QUBIT_REFERENCE_H
#define _QUBIT_REFERENCE_H
#include <vector>
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN

/**
 * @brief The position of the qubit is an expression
 * @ingroup QuantumMachine
 */
class QubitReference :public Qubit, public QubitReferenceInterface
{
private:
    std::shared_ptr<CExpr> m_cepr;
    std::vector<Qubit *> m_qvec;
public:
    inline PhysicalQubit* getPhysicalQubitPtr()  const
    {
        auto temp = m_cepr->get_val();
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

	inline std::shared_ptr<CExpr>  getExprPtr() { return m_cepr; }

    ~QubitReference()
    {
        m_cepr.reset();
    }

};
QPANDA_END
#endif 
