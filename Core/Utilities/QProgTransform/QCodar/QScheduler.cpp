#include <cstring>
#include <cmath>
#include <cctype>
#include <limits.h>
#include <time.h>
#include <random>
#include <chrono>
#include "Core/Utilities/QProgTransform/QCodar/QScheduler.h"
USING_QPANDA

#ifdef TESTING
// global varaibles for debugging
int global_debug_count = 0;
int debug_continuous_swap_count = 0;
#endif

int QScheduler::addLogicalQubit(int i, int j)
{
	auto next_index = map_list.size();
	map_list.push_back(i * device->getN() + j);
	return next_index;
}

void QScheduler::start()
{
	initMapping();  // register mapping to device

	mapped_result_gates.clear();
	std::vector<std::list<LogicalGate>> handle_logical_gates;
	std::list<LogicalGate> slicer_gates;

	std::vector<std::vector<LogicalGate>> handle_barriers;
	std::vector<LogicalGate> barrier_gates;

	for (auto gate : logical_gate_list)
	{
		if (gate.gate_type != GateType::BARRIER_GATE)
		{
			if (!barrier_gates.empty())
				handle_barriers.push_back(barrier_gates);

			barrier_gates.clear();
			slicer_gates.push_back(gate);
		}
		else
		{

			if (!slicer_gates.empty())
				handle_logical_gates.push_back(slicer_gates);

			slicer_gates.clear();
			barrier_gates.push_back(gate);
		}
	}
	if (!barrier_gates.empty())
		handle_barriers.push_back(barrier_gates);
	if (!slicer_gates.empty())
		handle_logical_gates.push_back(slicer_gates);

	for (int i = 0; i < handle_logical_gates.size(); i++)
	{
		logical_gate_list = handle_logical_gates[i];
		while (!(logical_gate_list.empty() && candidate_gates.empty()))
		{
			// start with zero priority on device
			// which is guaranteed in the constructor of device
			// reset priorities to 0 after used
			bool deadlock = true;
			selectCandidateGates(); // select CF gates
			// traverse candidate gates
			for (auto it = candidate_gates.begin(); it != candidate_gates.end(); )
			{
				auto &gate = *it;
				if (isImmediateGate(gate))
				{ // apply it if is an immediate gate
					if (isAllQubitFree(gate))
					{
						PhysicsGate ph_gate;
						getPhysicsGate(gate, ph_gate);
						launch(ph_gate);

#ifdef TESTING
						sum_min_route_len += gate.min_route_len;
						assert(gate.min_route_len >= 0);
						if (gate.isSingleQubitGate())
							sum_rx_route_len += gate.route_len;
						else
							sum_cnot_route_len += gate.route_len;

						max_route_len = std::max(max_route_len, gate.route_len);
#endif
						deadlock = false;
						it = candidate_gates.erase(it);  // erase such element
						if (it == candidate_gates.end())
							break;
						continue;   // `it` is already updated by `it = candidate_gates.erase(it);`
					}
					else
					{
						// immediate gate but can not apply now (not lock free)
						// participate in calculation of priorities
						if (!gate.isSingleQubitGate())
						{
							// a double-qubit gate may change to a non-immediate gate
							addPriority(gate);
						}
					}
				}
				else
				{    // non-immediate gate, we can calculate the prioritied and add candidate SWAPs
					// gate must be double qubit gate
					addPriority(gate);
				}
				it++;
			}
			// finished priorities set up
			if (!candidate_swaps.empty())
			{
				// there are candidate swaps
				while (true)
				{
					// select the SWAP with highest priority
					PhysicsGate swap_gate;
					if (!getHighestPrioritySWAP(swap_gate, false))
					{
						// no SWAP selected
						// check deadlock
						if (deadlock && device->isAllQubitFree())
						{
							getHighestPrioritySWAP(swap_gate, true);
							deadlock = false;
						}
						else
						{
							// no deadlock
							break;
						}
					}
					// apply SWAP
					launch(swap_gate);
					deadlock = false;
					// update priorities
					//#warning VERY SLOW METHOD USED
					// TODO: I can optimize the method here... to make it faster
					candidate_swaps.clear();
					for (auto &gate : candidate_gates)
					{
						if (!gate.isSingleQubitGate())
						{
							addPriority(gate);
						}
					}
				}
				// finished routing, clear all priorities
				candidate_swaps.clear();
			}
			device->nextCycle();
		}

		if (i < handle_barriers.size())
		{
			for (auto iter : handle_barriers[i])
			{
				PhysicsGate ph_gate;
				getPhysicsGate(iter, ph_gate);
				launch(ph_gate);
			}
		}
	}
}

bool QScheduler::getHighestPrioritySWAP(PhysicsGate &gate, bool deadlock)
{
	std::pair<int, double> max_priority(deadlock ? -1 : 0, INT_MAX);
	int best_i1 = -1, best_j1 = -1, best_i2 = -1, best_j2 = -1;
	for (auto it = candidate_swaps.begin(); it != candidate_swaps.end(); it++)
	{
		auto &swp = it->first;
		auto &priority = it->second;
		int i1 = swp.first / device->getN();
		int j1 = swp.first % device->getN();
		int i2 = swp.second / device->getN();
		int j2 = swp.second % device->getN();

		assert(device->isNearBy(i1, j1, i2, j2));

		if (priority > max_priority && device->canSwap(i1, j1, i2, j2))
		{
			best_i1 = i1;
			best_j1 = j1;
			best_i2 = i2;
			best_j2 = j2;
			max_priority = priority;
		}
	}
	if (max_priority.first <= 0)
	{
		if (!deadlock)
			return false;

		assert(max_priority.first == 0);
	}

	gate.type = "SWAP";
	gate.is_apply_swap = true;
	gate.is_dagger = false;
	gate.gate_type = GateType::SWAP_GATE;
	gate.i1 = best_i1;
	gate.j1 = best_j1;
	gate.i2 = best_i2;
	gate.j2 = best_j2;
	return true;
}

void QScheduler::addPriority(const LogicalGate &gate)
{
	assert(!gate.isSingleQubitGate());
	int i1, j1, i2, j2;     // pos of qubits of gate
	getMappedPosition(gate.c, i1, j1);
	getMappedPosition(gate.t, i2, j2);
	int dist0 = device->getDistance(i1, j1, i2, j2);    // distance of qubits of gate
	int hv_diff = 0;        // hd-vd difference
	if (device->isSimpleGridDevice())
	{
		hv_diff = getHDVDDiff(i1, j1, i2, j2);
	}
	for (auto &q : device->getQubit(i1, j1).nearbyQubits)
	{
		if (q.first == i2 && q.second == j2)
			continue;

		int dist = device->getDistance(q.first, q.second, i2, j2);
		double fine = 0;
		if (device->isSimpleGridDevice())
		{
			fine = hv_diff - getHDVDDiff(q.first, q.second, i2, j2);
		}

		if (!physics_qubit_error.empty())
		{
			int q1 = q.first * device->getN() + q.second;
			int q2 = i1 * device->getN() + j1;
			fine = (1 - physics_qubit_error[q1][q2]);
		}

		addPriorityForSwap(q.first, q.second, i1, j1, dist0 - dist, fine);
	}

	for (auto &q : device->getQubit(i2, j2).nearbyQubits)
	{
		if (q.first == i1 && q.second == j1)
			continue;
		int dist = device->getDistance(i1, j1, q.first, q.second);
		int fine = 0;

		if (device->isSimpleGridDevice())
		{
			fine = hv_diff - getHDVDDiff(i1, j1, q.first, q.second);
		}
		addPriorityForSwap(i2, j2, q.first, q.second, dist0 - dist, fine);
	}
}

// list of know gates
#define DEF_GATE2(G)    G##1, G##2,
#define DEF_GATE(G)     G,
#define ALIAS_GATE(ALS, G)
#define COMMUTE(G1, G2)
enum gate_enum {
	none,
#include "Core/Utilities/QProgTransform/QCodar/commuting.def"
	other
};
#undef DEF_GATE2
#define DEF_GATE2(G)
#undef DEF_GATE
#define DEF_GATE(G)

static unsigned int commuting_table[gate_enum::other + 1];

void QScheduler::loadCommutingTable()
{
	commuting_table[gate_enum::none] = 0xFFFFFFFF;

#undef COMMUTE
#define COMMUTE(G1, G2) commuting_table[gate_enum::G1] |= 1 << gate_enum::G2;
#include "Core/Utilities/QProgTransform/QCodar/commuting.def"
#undef COMMUTE
#define COMMUTE(G1, G2)
	commuting_table[gate_enum::other] = 0;
}

static void getGtGc(QScheduler::LogicalGate &logical_gate, gate_enum &gt, gate_enum &gc)
{
	gt = gate_enum::other;
	gc = gate_enum::none;
	const char *pgate = logical_gate.gate.c_str();

#undef ALIAS_GATE
#define ALIAS_GATE(ALS, G)  if(strcmp(logical_gate.gate.c_str(), #ALS) == 0) { pgate = #G; }
#include "Core/Utilities/QProgTransform/QCodar/commuting.def"
#undef ALIAS_GATE
#define ALIAS_GATE(ALS, G)

#undef DEF_GATE2
#undef DEF_GATE
#define DEF_GATE2(G)        if(strcmp(pgate, #G) == 0) { gt = gate_enum::G##2; gc = gate_enum::G##1; }
#define DEF_GATE(G)         if(strcmp(pgate, #G) == 0) { gt = gate_enum::G; }
#include "Core/Utilities/QProgTransform/QCodar/commuting.def"
#undef DEF_GATE2
#undef DEF_GATE
#undef ALIAS_GATE
#undef COMMUTE
}

void QScheduler::selectCandidateGates()
{
	size_t qcount = getLogicalQubitCount();
	unsigned int *state = new unsigned int[qcount];
	memset(state, 0x0, qcount * sizeof(unsigned int));

	for (auto &logical_gate : candidate_gates)
	{
		gate_enum gt, gc;
		getGtGc(logical_gate, gt, gc);
		state[logical_gate.t] |= 1 << gt;
		if (logical_gate.c >= 0)
			state[logical_gate.c] |= 1 << gc;
	}
	int _temp_count = 0;
	for (auto it = logical_gate_list.begin(); it != logical_gate_list.end();)
	{
		auto &logical_gate = *it;
		gate_enum gt, gc;
		getGtGc(logical_gate, gt, gc);
		// now，gt and gc is the type of the gates
		if ((~(commuting_table[gt]) & (state[logical_gate.t])) == 0 &&
			(logical_gate.c < 0 || (~(commuting_table[gc]) & (state[logical_gate.c])) == 0))
		{
			// commuting
			// insert it into candidate_gates
#ifdef TESTING
			if (!it->isSingleQubitGate())
			{
				int i1, j1, i2, j2;
				getMappedPosition(it->c, i1, j1);
				getMappedPosition(it->t, i2, j2);
				it->min_route_len = std::abs(i1 - i2) + std::abs(j1 - j2) - 1;
			}
#endif
			candidate_gates.splice(candidate_gates.end(), logical_gate_list, it++);
			_temp_count = 0;
		}
		else
		{
			++it;
			++_temp_count;
			if (_temp_count > device->getM() * device->getN() * 4)
				break;
		}
		state[logical_gate.t] |= 1 << gt;
		if (logical_gate.c >= 0)
			state[logical_gate.c] |= 1 << gc;
	}

	delete[] state;
}

void QScheduler::initMapping(void)
{
	for (size_t k = 0; k < getLogicalQubitCount(); k++)
	{
		int i, j;
		this->getMappedPosition(k, i, j);
		auto &q = device->getQubit(i, j).map_to;
		assert(q == -1);
		q = k;
	}
	swap_gate_count = 0;
	gate_count = 0;
#ifdef TESTING
	sum_min_route_len = 0;
	sum_rx_route_len = 0;
	sum_cnot_route_len = 0;
	max_route_len = 0;
#endif
}

void QScheduler::launch(QScheduler::PhysicsGate &gate)
{
#ifdef TESTING
	global_debug_count++;
	if (gate.isSwapGate())
	{
		debug_continuous_swap_count++;
		if (debug_continuous_swap_count > 20)
		{
			std::cerr << "!!! WARNING: " << global_debug_count << std::endl;
			assert(false);
		}
	}
	else
	{
		debug_continuous_swap_count = 0;
	}
#endif
	gate_count++;
	GateInfo gate_info;
	gate_info.is_dagger = gate.is_dagger;
	gate_info.param = gate.param;
	gate_info.type = gate.gate_type;
	gate_info.gate_name = gate.type;
	gate_info.barrier_id = gate.barrier_id;
	if (gate.isSwapGate())
	{
		// update the map table
		int q1 = device->getQubit(gate.i1, gate.j1).map_to;
		int q2 = device->getQubit(gate.i2, gate.j2).map_to;
		if (q1 != -1)
			map_list[q1] = gate.i2 * device->getN() + gate.j2;
		if (q2 != -1)
			map_list[q2] = gate.i1 * device->getN() + gate.j1;
#ifdef TESTING
		for (auto &gate : candidate_gates)
		{
			if ((gate.c != -1 && (gate.c == q1 || gate.c == q2))
				|| (gate.t == q1 || gate.t == q2))
			{
				gate.route_len++;
			}
		}
#endif
		// operate on device
		gate_info.control = gate.i1 * device->getN() + gate.j1;
		gate_info.target = gate.i2 * device->getN() + gate.j2;
		swap_gate_count++;
		device->applySwap(gate.i1, gate.j1, gate.i2, gate.j2/*, gate.gate_type*/);

		if (!physics_qubit_error.empty())
			double_gate_error_rate += 3 * physics_qubit_error[gate_info.control][gate_info.target];
	}
	else if (gate.isControlGate())
	{
		device->applyDoubleGate(gate.i1, gate.j1, gate.i2, gate.j2/*, gate.gate_type*/);
		gate_info.control = gate.i1 * device->getN() + gate.j1;
		gate_info.target = gate.i2 * device->getN() + gate.j2;

		if (!physics_qubit_error.empty())
			double_gate_error_rate += physics_qubit_error[gate_info.control][gate_info.target];  //  need to consider the double gate type
	}
	else
	{
		gate_info.control = -1;
		gate_info.target = gate.i1 * device->getN() + gate.j1;
		device->applySingleGate(gate.i1, gate.j1/*, gate.gate_type*/);
	}

	mapped_result_gates.push_back(gate_info);
}

bool QScheduler::addLogicalQubits(int count, bool is_order)
{
	std::vector<int> random_map(count, 0);
	int m = device->getM();
	int n = device->getN();
	if (is_order == true)
	{
		for (int k = 0; k < count; k++)
		{
			int i = k / n;
			int j = k % n;
			addLogicalQubit(i, j);
		}
		return true;
	}

	if (physics_qubit_error.empty())
	{
		// create a random sequence
		srand(time(nullptr));
		for (int i = 0; i < count; i++)
		{
			random_map[i] = i;
		}

		for (int i = count - 1; i > 0; i--)
		{
			std::swap(random_map[i], random_map[rand() % i]);
		}
		int mid_i = m / 2, mid_j = n / 2;
		if (m * n < count) {
			return false;
		}
		// try to map in a square
		int a = std::ceil(std::sqrt(count));
		if (a > m)
		{
			n = std::ceil((float)count / m);
		}
		else if (a > n)
		{
			m = std::ceil((float)count / n);
		}
		else
		{
			m = n = a;
		}

		int i0 = mid_i - m / 2;
		int j0 = mid_j - n / 2;
		for (int k = 0; k < count; k++)
		{
			int mp = random_map[k];
			int i = i0 + mp / n;
			int j = j0 + mp % n;
			addLogicalQubit(i, j);
		}
	}
	else
	{
		std::vector<double> fidelity;
		std::vector<int > valid_physics;

		std::vector<int > unused_phy;
		std::vector<int > unused_loc;

		for (int i = 0; i < physics_gate_fidelity.size(); i++)
		{
			if (physics_gate_fidelity[i] < 1e-15)
			{
				unused_phy.push_back(i);
				continue;
			}

			fidelity.push_back(physics_gate_fidelity[i]);
			valid_physics.push_back(i);
		}

		std::vector< std::pair<int, int> > apply_vec;
		for (auto val : logical_qubit_apply_degree)
		{
			apply_vec.push_back({ val.second, val.first });
		}

		for (int i = 0; i < count; i++)
		{
			auto iter = logical_qubit_apply_degree.find(i);
			if (iter == logical_qubit_apply_degree.end())
			{
				unused_loc.push_back(i);
			}
		}

		auto sort_func = [](std::pair<int, int> lh, std::pair<int, int> rh) {return lh.first > rh.first; };
		sort(apply_vec.begin(), apply_vec.end(), sort_func);

		std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
		for (auto val : apply_vec)
		{
			std::discrete_distribution<> distribution(fidelity.begin(), fidelity.end());
			int id = distribution(generator);
			random_map[val.second] = valid_physics[id];
			fidelity.erase(fidelity.begin() + id);
			valid_physics.erase(valid_physics.begin() + id);
		}

		unused_phy.insert(unused_phy.end(), valid_physics.begin(), valid_physics.end());
		for (int i = 0; i < unused_loc.size(); i++)
		{
			random_map[unused_loc[i]] = unused_phy.back();
			unused_phy.erase(unused_phy.end() - 1);
		}

		for (int k = 0; k < count; k++)
		{
			int mp = random_map[k];
			int i = mp / n;
			int j = mp % n;
			addLogicalQubit(i, j);
		}
	}
	return true;
}