/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QCircuitRewrite.h
Author: chenmingyu
Created in 2021/04/13

Classes for QCircuitRewrite.

*/
/*! \file QCircuitRewrite.h */
#ifndef QCIRCUITREWRITE_H
#define QCIRCUITREWRITE_H

#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/Tools/PraseExpressionStr.h"
#include "Core/Utilities/Tools/ThreadPool.h"
#include "Core/Core.h"

#include <thread>
QPANDA_BEGIN

class QCircuitRewrite {
private:
	CPUQVM m_qvm;
	QVec m_qv;
public:
	struct StructMatch {
		// the mapping from pattern(smaller graph) to graph
		std::map<int, int> core_1;
		// the mapping from graph to pattern(smaller graph)
		std::map<int, int> core_2;
		// T1 is the attribute of pattern
		std::set<int> T1_in;
		std::set<int> T1_out;
		// T2 is the attribute of graph
		std::set<int> T2_in;
		std::set<int> T2_out;
	};
	struct SemMatch {
		std::map<int, int> core_3;
		std::map<int, int> core_4;
		// whether the node on the graph matched or not
		std::set<int> p_qubit_tobematched;
		std::set<int> g_qubit_tobematched;
		std::map<int, double> angle_map;
	};
	struct MatchedSubgraph {
		std::map<int, int> match_vertices;
		std::map<int, int> match_qubits;
		std::map<int, double> match_angles;
		bool operator<(const MatchedSubgraph & rhs) const {
			return (match_vertices < rhs.match_vertices) | (match_vertices == rhs.match_vertices & match_qubits < rhs.match_qubits);
		}
	};
public:
	QCircuitRewrite() {
		m_qvm.init();
		m_qv = m_qvm.qAllocMany(10);
	}
	~QCircuitRewrite() {
		m_qvm.finalize();
	}

	int load_pattern_conf(const std::string& config_data);
	static double angle_str_to_double(const std::string angle_str)
	{
		double ret = 0.0;
		if (0 == strncmp(angle_str.c_str(), "theta_", 6))
		{
			ret = double(ANGLE_VAR_BASE) * atoi(angle_str.c_str() + 6);
		}
		else
		{
			ret = ParseExpressionStr().parse(angle_str);
		}
		return ret;
	}

	std::shared_ptr<QProgDAG> generator_to_dag(QCircuitGenerator& cir_gen);

	QProg replace_subgraph(std::shared_ptr<QProgDAG> g, QCircuitGenerator::Ref cir_gen);

	void PatternMatch(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph);
	void recursiveMatch(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph);

	bool feasibilityRules(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph, int n, int m, StructMatch & match);

	std::vector<std::set<uint32_t>> DAGPartition(std::shared_ptr<QProgDAG> graph, uint32_t par_num, uint32_t overlap);

	std::shared_ptr<QProg> circuitRewrite(QProg prog, uint32_t num_of_thread = 1);
public:
	StructMatch m_struct;
	SemMatch m_sem;
	std::set<int> matched;
	std::set<MatchedSubgraph> m_match_list;
public:
	threadPool m_thread_pool;
	static std::atomic<size_t> m_job_cnt;
	JsonConfigParam m_config_file;
	std::vector<std::pair<QCircuitGenerator::Ref, QCircuitGenerator::Ref>> m_optimizer_cir_vec;
};

/**
* @brief quantum-sub-circuit replacement by pattern matching algorithm
* @ingroup Utilities
* @param[in,out]  QProg&(or	QCircuit&) the source prog(or circuit)
* @param[in] const std::string& Pattern matching profile, it can be configuration file or configuration data, 
             which can be distinguished by file suffix,
			 so the configuration file must be end with ".json"
* @param[in] const uint32_t& Thread number
* @return     void
*/
void sub_cir_replace(QCircuit& src_cir, const std::string& config_data, const uint32_t& thread_cnt = 1);
void sub_cir_replace(QProg& src_prog, const std::string& config_data, const uint32_t& thread_cnt = 1);

QPANDA_END

#endif
