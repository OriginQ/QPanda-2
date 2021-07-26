#ifndef _QUANTUM_STATE_TOMOGRAPHY_H_
#define _QUANTUM_STATE_TOMOGRAPHY_H_

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"

QPANDA_BEGIN


class QuantumStateTomography
{
public:
    QuantumStateTomography();
    virtual ~QuantumStateTomography();

    template<typename T>
    std::vector<QProg> &combine_qprogs(const T &node, const QVec &qlist)
    {
        m_qlist = qlist;
        m_clist.clear();

        auto cbit_pool = OriginCMem::get_instance();
        for (auto &q : qlist)
        {
            m_clist.push_back(cbit_pool->cAlloc(q->get_phy_addr()));
        }
        _tomography_meas(node);
        return m_combine_progs;
    }

    template<typename T>
    std::vector<QProg> &combine_qprogs(const T &node, const std::vector<size_t> &qlist)
    {
        QVec qv;
        OriginQubitPool *qubit_pool = OriginQubitPool::get_instance();
        for (auto qaddr : qlist)
        {
            qv.push_back(qubit_pool->allocateQubitThroughVirAddress(qaddr));
        }
        return combine_qprogs(node, qv);
    }

    std::vector<QStat> exec(QuantumMachine *qm, size_t shots);
protected:
    template<typename T>
    void _tomography_meas(const T &node)
    {
        QProg tmp_prog;
        tmp_prog << node << BARRIER(m_qlist);
        m_combine_progs.assign(1, tmp_prog);
        for (auto iter = m_qlist.rbegin(); iter < m_qlist.rend(); iter++)
        {
            for (size_t i = 0; i < m_combine_progs.size(); i += 3)
            {
                auto cir_ry = deepCopy(m_combine_progs[i]);
                auto cir_rx = deepCopy(m_combine_progs[i]);
                m_combine_progs.insert(m_combine_progs.begin() + i + 1, cir_ry << RY(*iter, -PI / 2)); // RY(-PI/2)
                m_combine_progs.insert(m_combine_progs.begin() + i + 2, cir_rx << RX(*iter, PI / 2));   // RX(PI/2)
            }
        }

        for (size_t i = 0; i < m_combine_progs.size(); i++)
        {
            m_combine_progs[i] << MeasureAll(m_qlist, m_clist);
        }

        return;
    }
    void _get_s();
private:
    QVec m_qlist;
    std::vector<ClassicalCondition> m_clist;
    std::vector<QProg> m_combine_progs;
    std::vector<double> m_s;
    std::vector<std::map<std::string, double>> m_prog_results;
};



QPANDA_END

#endif
