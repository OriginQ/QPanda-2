/*! \file SingleAmplitudeQVM.h */
#ifndef  _SINGLEAMPLITUDE_H_
#define  _SINGLEAMPLITUDE_H_
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
#include "include/Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
QPANDA_BEGIN

/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/


/**
* @class SingleAmplitudeQVM
* @brief Quantum machine for single amplitude simulation
* @ingroup QuantumMachine
*/
class SingleAmplitudeQVM : public QVM, public TraversalQProg
{
public:
    SingleAmplitudeQVM();
    ~SingleAmplitudeQVM() {};

   /**
   * @brief  Init  the quantum  machine environment
   * @return     void  
   * @note   use  this at the begin
   */
   void init();
   void traversalAll(AbstractQuantumProgram*);
   void traversal(AbstractQGateNode*);

   /**
   * @brief  Load the quanyum program
   * @param[in]  QProg&  the reference to a quantum program 
   * @return     void  
   */
   void run(QProg&);

   /**
   * @brief  Get the quantum state of QProg
   * @return     QStat   quantum state
   * @exception   run_fail   pQProg is null
   */
   QStat getQStat();

   /**
   * @brief  PMeasure index
   * @param[in]  size_t  Abstract Quantum program pointer
   * @return     double  
   * @exception    Abstract Quantum program pointer is a nullptr
   * @note
   */
   double PMeasure_index(size_t);
   vector<double> PMeasure(QVec, int);
   std::vector<std::pair<size_t, double>> PMeasure(int);

   std::vector<double> getProbList(QVec, int);
   std::vector<double> probRunList(QProg &, QVec, int);

   std::map<std::string, double> getProbDict(QVec, int);
   std::map<std::string, double> probRunDict(QProg &, QVec, int);

   std::vector<std::pair<size_t, double>> getProbTupleList(QVec, int);
   std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec, int);

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
