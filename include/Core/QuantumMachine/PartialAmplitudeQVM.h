/*! \file PartialAmplitudeQVM.h */
#ifndef  _PARTIALAMPLITUDE_H_
#define  _PARTIALAMPLITUDE_H_
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "Core/VirtualQuantumProcessor/PartialAmplitude/PartialAmplitudeGraph.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Compiler/QRunesToQProg.h"
#include "Core/Utilities/Tools/Traversal.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @class PartialAmplitudeQVM   
* @ingroup QuantumMachine
* @see QuantumMachine
* @brief Quantum machine for partial amplitude simulation
* @ingroup QuantumMachine
*/
class PartialAmplitudeQVM : public QVM,
                            public TraversalInterface<>,
                            public MultiPrecisionMachineInterface
{
public:
    PartialAmplitudeQVM();
    ~PartialAmplitudeQVM();

    void init();


    /* new interface */
    /**
    * @brief  PMeasure by binary index
    * @param[in]  std::string  binary index
    * @return     qstate_type double
    * @note  example: PMeasure_bin_index("0000000000")
    */
    qstate_type pMeasureBinIndex(std::string str)
    {
        return PMeasure_bin_index(str);
    }

    /**
    * @brief  PMeasure by decimal  index
    * @param[in]  std::string  decimal index
    * @return     qstate_type double
    * @note  example: PMeasure_dec_index("1")
    */
    qstate_type pMeasureDecIndex(std::string str)
    {
        return PMeasure_dec_index(str);
    }

    /*will delete*/
    stat_map getQState();
    qstate_type PMeasure_bin_index(std::string);
    qstate_type PMeasure_dec_index(std::string);
    prob_map PMeasure(std::string);
    prob_map PMeasure(QVec, std::string);
    prob_map getProbDict(QVec, std::string);
    prob_map probRunDict(QProg &, QVec, std::string);
    prob_map PMeasureSubSet(QProg &, std::vector<std::string>);


    void run(std::string sFilePath)
    {
        auto prog = CreateEmptyQProg();
        transformQRunesToQProg(sFilePath, prog, this);
        run(prog);
    }

    template <typename _Ty>
    void run(_Ty &node)
    {
        m_prog_map->init(getAllocateQubit());

        traversal(node);
        m_prog_map->traversalQlist(m_prog_map->m_circuit);
        if (0 == m_prog_map->getMapVecSize())
        {
            m_prog_map->splitQlist(m_prog_map->m_circuit);
        }
    }

    /**
    * @brief    PMeasureSubSet
    * @param[in]  QProg  qubits vec
    * @param[in]  std::vector<std::string>
    * @return     prob_map std::map<std::string, qstate_type>
    * @note  output example: <0000000110:0.000167552>
    */
    prob_map pMeasureSubset(QProg &, std::vector<std::string>);

    template <typename _Ty>
    void traversal(_Ty &node)
    {
        execute(node.getImplementationPtr(), nullptr);
    }

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>);

private:
    PartialAmplitudeGraph *m_prog_map;
    void getSubGraphStat(vector<vector<QStat>> &);
};

QPANDA_END
#endif

