/*! \file SingleAmplitudeQVM.h */
#ifndef  _SINGLEAMPLITUDE_H_
#define  _SINGLEAMPLITUDE_H_
#include "Core/Utilities/Uinteger.h"
#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
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

class SingleAmplitudeQVM : public QVM, public TraversalInterface<>
{
public:
    SingleAmplitudeQVM();
    ~SingleAmplitudeQVM() {};

   void init();

   /*will delete*/

   template <typename _Ty>
   qstate_type PMeasure_bin_index(_Ty &node, std::string bin_index)
   {
       run(node);
       return singleAmpBackEnd(bin_index);
   }


   template <typename _Ty>
   qstate_type PMeasure_dec_index(_Ty &node, std::string s_dec_index)
   {
       run(node);
       uint256_t dec_index(s_dec_index.c_str());
       return singleAmpBackEnd(integerToBinary(dec_index, getAllocateQubit()));
   }

   /* new interface */
   /**
   * @brief  PMeasure by binary index
   * @param[in]  std::string  binary index
   * @return     qstate_type double
   * @note  example: PMeasure_bin_index("0000000000")
   */
   template <typename _Ty>
   qstate_type pMeasureBinindex(_Ty &node, std::string s_dec_index)
   {
       run(node);
       uint256_t dec_index(s_dec_index.c_str());
       return singleAmpBackEnd(integerToBinary(dec_index, getAllocateQubit()));
   }

   /**
   * @brief  PMeasure by decimal  index
   * @param[in]  std::string  decimal index
   * @return     qstate_type double
   * @note  example: PMeasure_dec_index("1")
   */
   template <typename _Ty>
   qstate_type pMeasureDecindex(_Ty &node, std::string bin_index)
   {
       run(node);
       return singleAmpBackEnd(bin_index);
   }

   void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);
   void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>);
   void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>);
   void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>);
   void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>);
   void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>);


   /*will delete*/
   stat_map getQState();
   prob_map PMeasure(std::string);
   prob_map PMeasure(QVec, std::string);
   prob_map getProbDict(QVec, std::string);
   prob_map probRunDict(QProg &, QVec, std::string);

   template <typename _Ty>
   void run(_Ty &node)
   {
       static_assert(std::is_base_of<QNode, _Ty>::value, "node type is error");
       m_prog_map.clear();
       VerticeMatrix  *vertice_matrix = m_prog_map.getVerticeMatrix();
       vertice_matrix->initVerticeMatrix(getAllocateQubit());
       m_prog_map.setQubitNum(getAllocateQubit());
       traversal(node);
   }

   void run(std::string sFilePath)
   {
       auto prog = CreateEmptyQProg();
       transformQRunesToQProg(sFilePath, prog, this);
       run(prog);
   }

   template <typename _Ty>
   void traversal(_Ty &node)
   {
       static_assert(std::is_base_of<QNode, _Ty>::value, "node type is error");
       Traversal::traversalByType(node.getImplementationPtr(), nullptr, *this);
   }

private:
    QuantumProgMap m_prog_map;

    qstate_type singleAmpBackEnd(string bin_index);
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, bool)> > m_singleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, bool)>> m_doubleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, double, bool)>> m_singleAngleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, double, bool)>> m_doubleAngleGateFunc;
};

QPANDA_END
#endif
