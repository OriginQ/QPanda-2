/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgStored.h
Author: Wangjing
Created in 2018-8-4

Classes for saving QProg as binary file.

Update@2018-8-30
update comment

*/
/*! \file QProgStored.h */
#ifndef QPROGSTORED_H
#define QPROGSTORED_H

#include <stdio.h>
#include <fstream>
#include <istream>
#include <list>
#include <string>
#include <map>
#include "Core/Utilities/Compiler/QProgToQuil.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN

const unsigned short kUshortMax = 65535;
const int kCountMoveBit = 16;

/**
* @brief  Quantum Program Stored Node Type
*/
enum QProgStoredNodeType {
	QPROG_PAULI_X_GATE = 1u,
	QPROG_PAULI_Y_GATE,
	QPROG_PAULI_Z_GATE,
	QPROG_X_HALF_PI,
	QPROG_Y_HALF_PI,
	QPROG_Z_HALF_PI,
	QPROG_HADAMARD_GATE,
	QPROG_T_GATE,
	QPROG_S_GATE,
	QPROG_RX_GATE,
	QPROG_RY_GATE,
	QPROG_RZ_GATE,
	QPROG_U1_GATE,
	QPROG_U2_GATE,
	QPROG_U3_GATE,
	QPROG_U4_GATE,
	QPROG_CU_GATE,
	QPROG_CNOT_GATE,
	QPROG_CZ_GATE,
	QPROG_CPHASE_GATE,
	QPROG_ISWAP_GATE,
	QPROG_ISWAP_THETA_GATE,
	QPROG_SQISWAP_GATE,
	QPROG_SWAP_GATE,
	QPROG_GATE_ANGLE,
	QPROG_MEASURE_GATE,
	QPROG_QIF_NODE,
	QPROG_QWHILE_NODE,
	QPROG_CEXPR_CBIT,
	QPROG_CEXPR_OPERATOR,
	QPROG_CEXPR_CONSTVALUE,
	QPROG_CEXPR_EVAL,
	QPROG_CEXPR_NODE,
	QPROG_CONTROL,
	QPROG_CIRCUIT_NODE,
	QPROG_RESET_NODE,
	QPROG_I_GATE,

};

/**
* @class  QProgStored
* @ingroup Utilities
* @brief  Utilities class for quantum program stored to binary data
*/
class QProgStored : public TraversalInterface<>
{
    union DataNode
    {
        DataNode() {}
        DataNode(uint32_t uiData) : qubit_data(uiData) {}
        DataNode(float fData) : angle_data(fData) {}

        uint32_t qubit_data;
        float angle_data;
    };
    typedef std::vector<std::pair<uint32_t, DataNode>>      dataList_t;
    typedef std::map<int, std::string>                      gateMap_t;

    typedef std::map<std::string, int>                      operatorMap_t;
public:
  QProgStored(QuantumMachine *qm);
  ~QProgStored();

  /**
   * @brief  transform quantum  program
   * @param[in]  QProg& quantum  program
   * @return     void
   */
  void transform(QProg &prog);

  /**
    * @brief  Store quantum program data
    * @param[in]  const std::string& filename  
    * @return     void  
    */
  void store(const std::string &);

  /**
    * @brief  get quantum program binary data
    * @return     std::vector<uint8_t>   quantum program binary data vector
    */
  std::vector<uint8_t> getInsturctions();

  virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQNoiseNode> cur_node, std::shared_ptr<QNode> parent_node);
  virtual void execute(std::shared_ptr<AbstractQDebugNode> cur_node, std::shared_ptr<QNode> parent_node);

private:
	void transformQProgByTraversalAlg(QProg *prog);

    void transformQControlFlow(AbstractControlFlowNode *controlflow);
	void transformQGate(AbstractQGateNode *gate);
    void transformQMeasure(AbstractQuantumMeasure *measure);
	void transformQReset(AbstractQuantumReset *p_reset);

    void transformQIfProg(AbstractControlFlowNode *controlflow);
    void transformQWhileProg(AbstractControlFlowNode *controlflow);
    void transformCExpr(CExpr *cexpr);
    void transformClassicalProg(AbstractClassicalProg *cc_pro);
    void handleQGateWithOneAngle(AbstractQGateNode *gate);
    void handleQGateWithFourAngle(AbstractQGateNode *gate);
    void addDataNode(const QProgStoredNodeType &type, const DataNode &data, const bool &is_dagger = false);

    QProg m_QProg;
    uint32_t m_node_counter;
    uint32_t m_qubit_number;

    uint32_t m_cbit_number;
    dataList_t m_data_vector;
    operatorMap_t m_operator_map;
    QuantumMachine * m_quantum_machine;
};


/**
* @brief  Get quantum program binary data
* @ingroup Utilities
* @param[in]  QProg& quantum program  
* @param[in]  QuantumMachine& quantum 
* @return     std::vector<uint8_t>   quantum program binary data
*/
std::vector<uint8_t> transformQProgToBinary(QProg &prog, QuantumMachine *qm);

/*will delete*/

void storeQProgInBinary(QProg &prog, QuantumMachine *qm, const std::string &filename);


/* new interface */

/**
* @brief  Store quantum program in binary file
* @ingroup Utilities
* @param[in]  QProg& quantum program
* @param[in]  QuantumMachine*  quantum machine
* @param[in]  std::string&	binary filename
* @return     void
*/
void transformQProgToBinary(QProg &prog, QuantumMachine *qm, const std::string &filename);

/**
* @brief  Get quantum program binary data
* @ingroup Utilities
* @param[in]  QProg& quantum program
* @param[in]  QuantumMachine& quantum
* @return     std::vector<uint8_t>   quantum program binary data
*/
std::vector<uint8_t> convert_qprog_to_binary(QProg &prog, QuantumMachine *qm);

/**
* @brief  Store quantum program in binary file
* @ingroup Utilities
* @param[in]  QProg& quantum program
* @param[in]  QuantumMachine*  quantum machine
* @param[in]  std::string&	binary filename
* @return     void
*/
void convert_qprog_to_binary(QProg &prog, QuantumMachine *qm, const std::string &filename);

QPANDA_END

#endif // QPROGSTORED_H
