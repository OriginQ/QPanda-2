/*! \file PartialAmplitudeQVM.h */
#ifndef  _PARTIALAMPLITUDE_H_
#define  _PARTIALAMPLITUDE_H_
#include "include/Core/Utilities/Uinteger.h"
#include "include/Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/MergeMap.h"

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
                            public TraversalQProg,
                            public MultiPrecisionMachineInterface
{
public:
    /**
    * @brief  Init the quantum machine environment
    * @return     void
    * @note   use this at the begin
    */
    void init();


    /**
    * @brief  Load and parser Quantum Program
    * @param[in]  QProg &  Quantum Program
    * @return     void
    */
    void run(QProg&);

    /**
    * @brief  Load and parser Quantum Program by file
    * @param[in]  std::string  Quantum Program QRunes file path
    * @return     void
    */
    void run(std::string);


    PartialAmplitudeQVM();
    ~PartialAmplitudeQVM();


    /**
    * @brief  Get Quantum State 
    * @return   std::map<std::string, qcomplex_t>
    * @note  output example: <0000000000:(-0.00647209,-0.00647209)>
    */
    stat_map getQStat();


    /**
    * @brief  PMeasure by binary index
    * @param[in]  std::string  binary index
    * @return     qstate_type double
    * @note  example: PMeasure_bin_index("0000000000")
    */
    qstate_type PMeasure_bin_index(std::string);

    /**
    * @brief  PMeasure by decimal  index
    * @param[in]  std::string  decimal index
    * @return     qstate_type double
    * @note  example: PMeasure_dec_index("1")
    */
    qstate_type PMeasure_dec_index(std::string);


    /**
    * @brief  PMeasure
    * @param[in]  std::string  select max
    * @return     prob_map  std::map<std::string, qstate_type>
    */
    prob_map PMeasure(std::string);


    /**
    * @brief  PMeasure
    * @param[in]  QVec    qubits vec
    * @param[in]  std::string    select max
    * @return     prob_map   std::map<std::string, qstate_type>
    */
    prob_map PMeasure(QVec, std::string);

    /**
    * @brief  Get quantum state Probability dict
    * @param[in]  QVec  qubits vec
    * @param[in]  std::string   select max
    * @return     prob_map std::map<std::string, qstate_type>
    * @note  output example: <0000000110:0.000167552>
    */
    prob_map getProbDict(QVec, std::string);

    /**
    * @brief  Run and get quantum state Probability dict
    * @param[in]  QVec  qubits vec
    * @param[in]  std::string   select max
    * @return     prob_map std::map<std::string, qstate_type>
    * @note  output example: <0000000110:0.000167552>
    */
    prob_map probRunDict(QProg &, QVec, std::string);

    /**
    * @brief    PMeasureSubSet
    * @param[in]  QProg  qubits vec
    * @param[in]  std::vector<std::string>   
    * @return     prob_map std::map<std::string, qstate_type>
    * @note  output example: <0000000110:0.000167552>
    */
    prob_map PMeasureSubSet(QProg &prog, std::vector<std::string> subset_vec);
private:
    MergeMap *m_prog_map;

    void traversal(AbstractQGateNode *);
    void traversalAll(AbstractQuantumProgram *);
    void getSubGraphStat(vector<vector<QStat>> &);
};

QPANDA_END
#endif

