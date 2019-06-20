/*! \file SingleAmplitudeQVM.h */
#ifndef  _SINGLEAMPLITUDE_H_
#define  _SINGLEAMPLITUDE_H_
#include "include/Core/Utilities/Uinteger.h"
#include "include/Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/


/**
* @class SingleAmplitudeQVM
* @ingroup QuantumMachine
* @see QuantumMachine
* @brief Quantum machine for single amplitude simulation
* @ingroup QuantumMachine
*/

class SingleAmplitudeQVM : public QVM,
                           public TraversalQProg,
                           public MultiPrecisionMachineInterface
{
public:
    SingleAmplitudeQVM();
    ~SingleAmplitudeQVM() {};

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

   void traversal(AbstractQGateNode*);
   void traversalAll(AbstractQuantumProgram*);

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
   * @brief  get quantum state Probability dict
   * @param[in]  QVec  qubits vec
   * @param[in]  std::string   select max
   * @return     prob_map std::map<std::string, qstate_type>
   * @note  output example: <0000000110:0.000167552>
   */
   prob_map getProbDict(QVec, std::string);

   /**
   * @brief  run and get quantum state Probability dict
   * @param[in]  QVec  qubits vec
   * @param[in]  std::string   select max
   * @return     prob_map std::map<std::string, qstate_type>
   * @note  output example: <0000000110:0.000167552>
   */
   prob_map probRunDict(QProg &, QVec, std::string);

private:
    QProg m_prog;
    QuantumProgMap m_prog_map;

    std::map<size_t, std::function<void(QuantumProgMap &, size_t, bool)> > m_singleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, bool)>> m_doubleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, double, bool)>> m_singleAngleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, double, bool)>> m_doubleAngleGateFunc;
};

QPANDA_END
#endif
