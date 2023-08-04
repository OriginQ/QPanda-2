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

#ifndef __GRID_DEVICE_H__
#define __GRID_DEVICE_H__

#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

#define NOMINMAX
#undef max
#undef min
QPANDA_BEGIN

 // gate duration, using MACROs for quick prototype
#ifndef SWAP_TIME
#define SWAP_TIME   6
#endif
#ifndef DOUBLE_GATE_TIME
#define DOUBLE_GATE_TIME   2
#endif
#ifndef SINGLE_GATE_TIME
#define SINGLE_GATE_TIME     1
#endif

//#define  TESTING 

/**
* @class BasicGridDevice
* @ingroup Utilities
* @brief	A BasicGirdDevice is a device model that qubits put on nodes of a rectangular grid
*				It is a virtual class because we haven't dicided how qubits connecting with each other
*				For the simpliest situation, see SimpleGridDevice
*/
class BasicGridDevice
{
public:
	struct PhysicalQubit 
	{
		int map_to = -1;		/**< which logical qubit maps to it, -1 if no */
		int time;					/**< time when it become free again, the qubit lock  */
		std::vector<std::pair<int, int>> nearbyQubits;
	};

	/**
	 * @brief  constructor constructor
	 * @note m, n: the grid side length
	 */
	BasicGridDevice(int m, int n);

	/**
	 * @brief  clear all properties except gird side length,  re-initialize
	 */
	void clear();

	virtual ~BasicGridDevice();

	inline int getM() 
	{ 
		return m;
	}

	inline int getN() 
	{ 
		return n; 
	}

	/**
	 * @brief get the qubit on (i, j)
	 */
	inline auto &getQubit(int i, int j)
	{
		assert(i >= 0 && i < m);
		assert(j >= 0 && j < n);
		return qubits[i * n + j];
	}

	/**
	 * @brief Determine whether a double-qubit gate can be applied
	 */
	inline bool canApplyGate(int i1, int j1, int i2, int j2, int /* time */)
	{
		return getQubit(i1, j1).time <= current_time
			&& getQubit(i2, j2).time <= current_time;
	}

	/**
	 * @brief Determine whether a single-qubit gate can be applied
	 */
	inline bool canApplyGate(int i, int j, int /* time */) 
	{
		return getQubit(i, j).time <= current_time;
	}


	inline bool canSwap(int i1, int j1, int i2, int j2) 
	{
		if (isSupportSwapGate())
		{
			return canApplyGate(i1, j1, i2, j2, DOUBLE_GATE_TIME) && isNearBy(i1, j1, i2, j2);
		}
		else
		{
			return canApplyGate(i1, j1, i2, j2, SWAP_TIME) && isNearBy(i1, j1, i2, j2);
		}
	}


	/**
	 * @brief apply gate and comsume time
	 */
	inline void applySingleGate(int i, int j)
	{
		auto &t = getQubit(i, j).time;
		t += SINGLE_GATE_TIME;
		qubit_max_time = std::max(qubit_max_time, t);
	}

	/**
	 * @brief apply a double-qubit gate
	 */
	inline void applyGate(int i1, int j1, int i2, int j2, int time) 
	{
		auto &t1 = getQubit(i1, j1).time;
		auto &t2 = getQubit(i2, j2).time;
		assert(t1 <= current_time && t2 <= current_time);
		t1 = t2 = std::max(t1, t2) + time;
		qubit_max_time = std::max(qubit_max_time, t1);
	}

	inline void applyDoubleGate(int i1, int j1, int i2, int j2)
	{
		applyGate(i1, j1, i2, j2, DOUBLE_GATE_TIME);
	}

	inline void applySwap(int i1, int j1, int i2, int j2)
	{
        assert(isNearBy(i1, j1, i2, j2));
		if (isSupportSwapGate())
		{
			applyGate(i1, j1, i2, j2, DOUBLE_GATE_TIME);
		}
		else
		{
			applyGate(i1, j1, i2, j2, SWAP_TIME);
		}
		std::swap(getQubit(i1, j1).map_to, getQubit(i2, j2).map_to);
	}

	/**
	 * @brief Check if two qubits are adjacent, location of qubits are (i1, j1) and (i2, j2)
	 */
	virtual bool isNearBy(int i1, int j1, int i2, int j2) = 0;
	
	/**
	 * @brief  get distance+1 of two qubits
	 */

	virtual int getDistance(int i1, int j1, int i2, int j2) = 0;


	/**
	 * @brief map one qubit
	 * @param[in] int  dest: index of the logical qubit
	 * @param[in] int  i: location of the physical qubit
	 * @param[in] int  j: location of the physical qubit
	 */
	inline void map(int dest, int i, int j)
	{
		getQubit(i, j).map_to = dest;
	}

	/**
	 * @brief  reset the time
	 */
	inline void resetTime()
	{
		for (int i = 0; i < m * n; i++) {
			qubits[i].time = 0;
		}
	}

	/**
	 * @brief  go on next instruction cycle
	 */
	inline void nextCycle()
	{
		current_time++;
	}

	/**
	 * @brief  time when all qubits end up being busy
	 */
	inline int maxTime() 
	{
		return qubit_max_time;
	}

	inline int getTime() 
	{
		return current_time;
	}

	/**
	 * @brief  check if physical qubit on (i, j) free
	 */
	inline bool isQubitFree(int i, int j)
	{
		return getQubit(i, j).time <= current_time;
	}

	/**
	 * @brief  check if all physical qubits free
	 */
	inline bool isAllQubitFree()
	{
		return qubit_max_time <= current_time;
	}

	inline bool isSimpleGridDevice()
	{
		return is_simple_grid_device;
	}

	inline bool isSupportSwapGate()
	{
		return is_support_swap_gate;
	}

protected:
	bool is_simple_grid_device = false;		/**< can caculate H_fine if it is true */
	bool is_support_swap_gate = false;		/**< support SWAP gate */
	int m, n;													/**< m, n: the grid side length */
	int current_time;
	int qubit_max_time;								/**< time when all qubits end up being busy */
	BasicGridDevice::PhysicalQubit *qubits;
};



/**
* @class ExGridDevice
* @ingroup Utilities
* @brief	ExGirdDevice is an extension of the grid device
*				The qubit layout is the same to BasicGridDevice
*				But the adjacency between qubits can be different
*				for example, qubit at (i, j) may be not adjacent to (i, j + 1)
*				but(i1 ,j1) and (i2, j2) may be adjacent
*				e.g. ibm tokyo q20 and ibm Rochester q53
*				ExGirdDevice is a derived class of BasicGridDevice
*/
class ExGridDevice : public BasicGridDevice {
public:
	/**
	 * @brief  constructor constructor
	 * @param[in] m the grid side length
	 * @param[in] n  the grid side length
	 * @param[in] qpairs  set of adjacent qubit pairs
	 * @note	whose elements are like <q1, q2>£¬q1, q2 ¡Ê [0, m * n)
	 *				q1 is at (q1 / n, q1 % n)
	 *				so is q2
	 */
	ExGridDevice(int m, int n, std::vector<std::pair<int, int>> &qpairs);

	~ExGridDevice() override;

	bool isNearBy(int i1, int j1, int i2, int j2) override;

	int getDistance(int i1, int j1, int i2, int j2) override;

protected:
	int *dist_mat;
	int qcount;
};



/**
* @class SimpleGridDevice
* @ingroup Utilities
* @brief	 A derived class of BasicGridDevice
*				 The two qubits are adjacent when their Manhattan distance is 1.
*/
class SimpleGridDevice : public BasicGridDevice 
{
public:
	inline SimpleGridDevice(int m, int n)
		: BasicGridDevice(m, n)
	{
		is_simple_grid_device = true;
		is_support_swap_gate = true;
		for (int i = 0; i < m; i++) 
		{
			for (int j = 0; j < n; j++) 
			{
				if (j > 0)
					getQubit(i, j).nearbyQubits.emplace_back(i, j - 1);
				if (j < n - 1)
					getQubit(i, j).nearbyQubits.emplace_back(i, j + 1);
				if (i > 0)
					getQubit(i, j).nearbyQubits.emplace_back(i - 1, j);
				if (i < m - 1)
					getQubit(i, j).nearbyQubits.emplace_back(i + 1, j);
			}
		}
	}

	~SimpleGridDevice() override;

	bool isNearBy(int i1, int j1, int i2, int j2) override;

	int getDistance(int i1, int j1, int i2, int j2) override;

protected:
	inline SimpleGridDevice(int m, int n, bool init)
		: BasicGridDevice(m, n)
	{
		is_simple_grid_device = true;
		if (init)
		{
			for (int i = 0; i < m; i++) 
			{
				for (int j = 0; j < n; j++)
				{
					if (j > 0)
						getQubit(i, j).nearbyQubits.emplace_back(i, j - 1);
					if (j < n - 1)
						getQubit(i, j).nearbyQubits.emplace_back(i, j + 1);
					if (i > 0)
						getQubit(i, j).nearbyQubits.emplace_back(i - 1, j);
					if (i < m - 1)
						getQubit(i, j).nearbyQubits.emplace_back(i + 1, j);
				}
			}
		}
	}
};


/**
* @class UncompletedGridDevice
* @ingroup Utilities
* @brief	UncompletedGridDevice is a derived class of SimpleGridDevice
*				Compared with SimpleGridDevice, some qubits are NOT AVAILABLE, but the adjacent relation is the same
*				e.g. google Sycamore q54
*/
class UncompletedGridDevice : public SimpleGridDevice 
{
public:
	inline UncompletedGridDevice(int m, int n, const bool *available_qubits)
		: SimpleGridDevice(m, n, false) 
	{
		is_simple_grid_device = true;
		this->available_qubits = new bool[m * n];
		resetAvailableQubits(available_qubits);
	}

	void resetAvailableQubits(const bool *available_qubits);

	~UncompletedGridDevice() override;

	inline bool isQubitAvailable(int i, int j) 
	{
		return available_qubits[i * n + j];
	}

private:
	bool *available_qubits;
};

QPANDA_END

#endif // __GRID_DEVICE_H__

