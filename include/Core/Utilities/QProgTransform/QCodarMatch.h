/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QCodarMatch.h */
#ifndef _QCODARMATCH_H
#define _QCODARMATCH_H

#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QProgTransform/QCodar/QScheduler.h"
#include "Core/Utilities/QProgTransform/QCodar/GridDevice.h"
#include "Core/QuantumCircuit/QuantumGate.h"

QPANDA_BEGIN

/**
* @brief QCodar Grid Device type
*/
enum QCodarGridDevice
{
	IBM_Q20_TOKYO = 0,
	IBM_Q53,
	GOOGLE_Q54,
	SIMPLE_TYPE,
};

/**
* @class QCodarMatch
* @ingroup Utilities
* @brief A Contextual Duration-Aware Qubit Mapping (CODAR)
*/
class QCodarMatch : public TraversalInterface<bool&>
{
public:
	QCodarMatch(QuantumMachine * machine, QCodarGridDevice arch_type , int m,  int n);
	~QCodarMatch();
	
	void initScheduler(QCodarGridDevice arch_type, size_t qubits);

	void initGridDevice(QCodarGridDevice arch_type, int &m, int &n);

	/**
	* @brief  Mapping qubits in a quantum program
	* @param[in]  Qprog  quantum program
	* @param[in]  size_t   run_times  : the number of times  run the remapping
	* @param[in,out]  QVec  qubits vector
	* @param[out]  Qprog&  the mapped quantum program
	* @return   void
	**/
	void mappingQProg(QProg prog, size_t run_times, QVec &qv, QProg &mapped_prog);

	/**
	* @brief  build QProg by the mapping results
	* @param[in]  std::vector<GateInfo>  gates info vector
	* @param[in]  std::vector<int>  map vector
	* @param[out]  Qprog&  the mapped quantum program
	* @return   void
	**/
	void buildResultingQProg(const std::vector<GateInfo> resulting_gates, const std::vector<int>  map_vec,  QVec &q, QProg &prog);

	/**
	* @brief traversal quantum program and Parsing quantum program information
	* @param[in] QProg*  quantum program pointer
	* @return   void
	**/
	void traversalQProgParsingInfo(QProg *prog);

	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &);

private:

	QuantumMachine * m_qvm;

	QScheduler  *m_scheduler;
	BasicGridDevice *m_device;

	std::vector<GateInfo> m_original_gates;  /**< the original gates set */
	std::map<int, std::string>  m_gatetype;		/**< quantum gate type map */

	std::map<int, std::function<QGate(Qubit *)> > m_single_gate_func;
	std::map<int, std::function<QGate(Qubit *, double)> > m_single_angle_gate_func;
	std::map<int, std::function<QGate(Qubit *, Qubit*)> > m_double_gate_func;
	std::map<int, std::function<QGate(Qubit *, Qubit*, double)> > m_double_angle_gate_func;

private:
};

/**
* @brief   A Contextual Duration-Aware Qubit Mapping for V arious NISQ Devices
* @ingroup Utilities
* @param[in]  QProg  quantum program
* @param[in,out]  QVec  qubit  vector
* @param[in]  QuantumMachine*  quantum machine
* @param[in]  QCodarGridDevice   grid device type
* @param[in]  size_t   m : the length of the topology,  if SIMPLE_TYPE  you need to set the size
* @param[in]  size_t   n  : the  width of the topology,   if SIMPLE_TYPE  you need to set the size
* @param[in]  size_t   run_times  : the number of times  run the remapping, better parameters get better results
* @return    QProg   mapped  quantum program
* @note	 QCodarGridDevice : SIMPLE_TYPE  It's a simple undirected  topology graph, build a topology based on the values of m and n
*												    0 = 1 = 2 = 3  
*				    eg: m = 2, n = 4		¨U    ¨U    ¨U    ¨U
*													4 = 5 = 6 = 7
*											   
*/
QProg qcodar_match(QProg prog, QVec &qv, QuantumMachine * machine, 
	QCodarGridDevice arch_type = SIMPLE_TYPE, size_t m = 2, size_t n = 4, size_t run_times = 5);


QPANDA_END

#endif // !_QCODARMATCH_H