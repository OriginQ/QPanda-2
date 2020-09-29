/*
Copyright (c) 2020 Prof. Yu Zhang All Right Reserved.

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

#ifndef QSCHEDULER_H
#define QSCHEDULER_H

#include <list>
#include <vector>
#include <set>
#include <map>
#include "Core/Utilities/QProgTransform/QCodar/GridDevice.h"
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

/**
* @brief Save the parsed logical gate information
*/
struct GateInfo
{
	int target;
	int control;
	int type;
	bool is_dagger;
	std::vector<double> param;
	std::string gate_name;
	int barrier_id;
};

/**
* @class QScheduler
* @ingroup Utilities
* @brief CODAR algorithm, used for mapping calculations
*/
class QScheduler 
{
public:
	inline QScheduler(BasicGridDevice *device)
		: device(device)
	{}

	/**
	 * @brief   logical gate
	 */
	struct LogicalGate
	{
		int t;						/**< index of the target qubit*/
		int c;						/**< index of the control qubit, c = -1 for single qubit gate */ 
		std::string gate;   	/**< type of the gate */
		 
		int gate_type;    /**< these parameters can be encapsulated*/
		bool is_dagger;
		std::vector<double> param;
		int barrier_id;
#ifdef TESTING
		int min_route_len = 0;		/**< min route length, the distance of qubits when it inserted into candidate gate list*/
		int route_len = 0;				/**<the actual route length*/
#endif
		bool isSingleQubitGate() const 
		{
			return c == -1;
		}

		/**
		 * @brief   create a gate
		 */
		inline LogicalGate(std::string gate, int gate_type,  int c, int t, std::vector <double> param, int barrier_id,  bool is_dagger)
			:gate(gate), gate_type(gate_type), c(c), t(t), param(param), barrier_id(barrier_id),is_dagger(is_dagger) { }
	};

	/**
	 * @brief  physical gate
	 */
	struct PhysicsGate 
	{
		std::string type;
		int gate_type;
		bool is_dagger;
		std::vector<double> param;
		bool is_apply_swap;
		int i1, j1;     /**< (i1, j1) is the position for the first qubit parament of the gate */
		int i2, j2;     /**< (i2, j2) is the position for the first qubit parament of the two-qubit gate */
		int barrier_id;
		inline bool isSwapGate() 
		{
			return is_apply_swap == true;
		}

		inline bool isControlGate() 
		{
			return i2 != -1 && j2 != -1;
		}

	};

	void loadCommutingTable();

	/**
	 * @brief  get count of logical qubits
	 */
	inline size_t getLogicalQubitCount()
	{
		return map_list.size();
	}

	/**
	 * @brief  add a logical qubit by its position
	 * return the index of the logical qubit
	 */
	int addLogicalQubit(int i, int j);

	/**
	 * @brief add several logical qubits and make initial mapping automaticly
	 * @param[in] int count of qubits
	 * @return bool true if success
	 * @note MUST ENSURE YOU HAVEN"T MAPPED ANY QUBIT BEFORE CALLING
	 */
	bool addLogicalQubits(int count, bool is_order = false);

	/**
	 * @brief add a (logical) single qubit gate
	 */
	inline void addSingleQubitGate(std::string gate, int gate_type, int t, std::vector <double> param ,int barrier_id =-1,  bool is_dagger = false)
	{
		logical_gate_list.emplace_back(gate, gate_type, -1, t, param, barrier_id,  is_dagger);
	}

	/**
	 * @brief add a logical double qubit gate
	*/
	inline void addDoubleQubitGate(std::string gate, int gate_type, int c, int t, std::vector <double> param , bool is_dagger = false)
	{
		logical_gate_list.emplace_back(gate, gate_type, c, t, param, -1,  is_dagger);
	}

	/**
	 * @brief start remapping (call it after intialization) , The main entry of the CODAR remapper
	 */
	void start();


	inline void setQubitFidelity(std::map<int, int > degree, std::vector<double> fidelity, std::vector< std::vector<double > > error_rate)
	{
		logical_qubit_apply_degree = degree;
		physics_gate_fidelity = fidelity;
		physics_qubit_error = error_rate;
	}
private:
	/**
	 * @brief create the candidate gate set
	 */
	void selectCandidateGates();

	/**
	 * @brief get a SWAP with highest priority in all possible routing SWAPs which could be apply in parallel now,
	 * @return bool false if no SWAP with positive priority found
	 * @note if deadlock == true, SWAP with zero priority may be ok, too.
	 */
	bool getHighestPrioritySWAP(PhysicsGate &gate, bool deadlock);

	/**
	 * @brief launch a physical gate (to the device), change the statement and output
	 */
	void launch(PhysicsGate &gate);

	/**
	 * @brief call this function after input initial mapping.
	 * @note this function will initialize the device
	 *				register mapping infomation to device
	 *				may cause error if you call the function for more than once or the initial mapping has conflicts
	 *				must ensure the mapping infomation in the device is empty
	 */
	void initMapping(void);


	/**
	 * @brief get the physical position of the logical qubit
	 * @param[in] int index of the logical qubit
	 * @param[out] int&  i:position of the physical qubit
	 * @param[out] int&  j:position of the physical qubit
	 */
	inline void getMappedPosition(int logical, int &i, int &j)
	{
		int x = map_list[logical];
		i = x / device->getN();
		j = x % device->getN();
	}

	/**
	 * @brief check whether a gate is immediate gate or not
	 */
	inline bool isImmediateGate(LogicalGate &gate)
	{
		if (gate.isSingleQubitGate()) 
		{
			// assume all 1-qubit gates are immediate
			return true;
		}
		else
		{
			int i1, j1, i2, j2;
			getMappedPosition(gate.c, i1, j1);
			getMappedPosition(gate.t, i2, j2);
			return device->isNearBy(i1, j1, i2, j2);
		}
	}

	/**
	 * @brief check whether the logical qubit is free or not
	 */
	inline bool isLogicalQubitFree(int logical_qubit)
	{
		int i, j;
		getMappedPosition(logical_qubit, i, j);
		return device->isQubitFree(i, j);
	}

	/**
	 * @brief check whether all qubits of the logical gate are free or not
	 */
	inline bool isAllQubitFree(LogicalGate &gate) 
	{
		return isLogicalQubitFree(gate.t) && 
			(gate.isSingleQubitGate() || isLogicalQubitFree(gate.c));
	}

	/**
	 * @brief get the corresponding physical gate of a logical gate
	 */
	inline void getPhysicsGate(LogicalGate &logical_gate, PhysicsGate &physics_gate) 
	{
		int it, jt, ic, jc;
		getMappedPosition(logical_gate.t, it, jt);
		if (logical_gate.isSingleQubitGate())
		{
			physics_gate.i1 = it;
			physics_gate.j1 = jt;
			physics_gate.i2 = -1;
			physics_gate.j2 = -1;
		}
		else 
		{
			getMappedPosition(logical_gate.c, ic, jc);
			physics_gate.i1 = ic;
			physics_gate.j1 = jc;
			physics_gate.i2 = it;
			physics_gate.j2 = jt;
		}
		physics_gate.is_apply_swap = false;
		physics_gate.type = logical_gate.gate;
		physics_gate.is_dagger = logical_gate.is_dagger;
		physics_gate.param = logical_gate.param;
		physics_gate.gate_type = logical_gate.gate_type;
		physics_gate.barrier_id = logical_gate.barrier_id;
	}

	/**
	 * @brief update the priority (both H_main and H_fine) of a SWAP by a logical gate
	 *	@note if the SWAP is not in candidate SWAP list, insert it into at first.
	 */
	void addPriority(const LogicalGate &gate);

	inline int getHDVDDiff(int i1, int j1, int i2, int j2)
	{
		return std::abs(std::abs(i1 - i2) - std::abs(j1 - j2));
	}

	inline void addPriorityForSwap(int i1, int j1, int i2, int j2, int main, double fine)
	{
		int q1 = i1 * device->getN() + j1;
		int q2 = i2 * device->getN() + j2;
		if (q1 > q2) 
		{
			std::swap(q1, q2);
		}
		auto swp = std::pair<int, int>(q1, q2);
		if (candidate_swaps.count(swp) == 0) 
		{
			candidate_swaps[swp] = std::pair<int, double>(main, fine);
		}
		else
		{
			candidate_swaps[swp].first += main;
			candidate_swaps[swp].second += fine;
		}
	}
	
public:
	std::vector<int> map_list;							/**< the mapping table from logical qubits to physical qubits, see getMappedPosition */
	BasicGridDevice *device;							/**<  the device */
	std::vector<GateInfo> mapped_result_gates;			/**< the output  */
	std::list<LogicalGate> logical_gate_list;	/**< the list of logical gates (the whole circuit) */
	std::list<LogicalGate> candidate_gates;    /**< the set of Commutative Forward gates */
	int gate_count = 0;										/**< count of gates launched */

	int swap_gate_count = 0; 

	// q1, q2: physical qubits to SWAP
	// candidate_swaps[std::pair(q1, q2)] = std::pair(H_main, H_fine)
	std::map<std::pair<int, int>, std::pair<int, double>>  candidate_swaps;   /**< maps from candidate swaps to their heuristic costs */

	std::map<int, int >  logical_qubit_apply_degree;
	std::vector<double> physics_gate_fidelity;
	std::vector< std::vector<double > > physics_qubit_error;
	double double_gate_error_rate = 0;
#ifdef TESTING
	int sum_min_route_len = 0;
	int sum_rx_route_len = 0;
	int sum_cnot_route_len = 0;
	int max_route_len = 0;
#endif

};

QPANDA_END
#endif // QSCHEDULER_H