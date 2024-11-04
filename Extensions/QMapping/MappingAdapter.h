#pragma once

#include <vector>
#include <variant>

#include "OBMTQMapping.h"
#include "SabreQMapping.h"

template<typename T, typename... Args>
void extract_params(std::vector<std::variant<bool, uint32_t>>& param_list, T first, Args... args)
{
    if constexpr (std::is_integral_v<T>) {
        param_list.push_back(static_cast<uint32_t> (first));
    }
    else if constexpr (std::is_same_v<T, bool>) {
	    param_list.push_back(first);
    }
    else {
        param_list.clear();
        return;
    }
	if constexpr (sizeof...(args) > 0) {
		extract_params(param_list, args...);
	}
	return;
}

template<typename T, typename C, typename... Args>
std::map<double, std::vector<QPanda::Mapping>, std::greater<double>>
select_best_qubits_blocks(QPanda::QProg src_prog, QPanda::QuantumMachine* qvm, const C& config, Args... args)
{
    QPanda::QProg prog = deepCopy(src_prog);
	flatten(prog);
    map_to_continues_qubits(prog, qvm);

    QPanda::QVec used_qv;
    /* Get all the used  quantum bits in the input prog */
    get_all_used_qubits(prog, used_qv);
    prog.insertQNode(prog.getHeadNodeIter(), std::dynamic_pointer_cast<QPanda::QNode>(BARRIER(used_qv).getImplementationPtr()));

    QPanda::RemoveMeasureNode measure_cutter;
    measure_cutter.remove_measure(prog);

	QPanda::ArchGraph::sRef graph;
	if constexpr (std::is_same_v<std::decay_t<C>, QPanda::QMappingConfig>) {
		graph = QPanda::OptBMTQAllocator::build_arch_graph(config);
	}
	else if constexpr (std::is_same_v<std::decay_t<C>, std::tuple<std::vector<std::vector<double>>, std::vector<uint32_t>>>) {
		auto &[matrix_connect, phy_partition] = config;
		graph = QPanda::OptBMTQAllocator::build_arch_graph(matrix_connect, phy_partition);
	}
	else {
		throw std::invalid_argument("Invalid config type");
	}

	//std::unique_ptr<AbstractQubitMapping> allocator;

	std::vector<std::variant<bool, uint32_t>> param_list;
	extract_params(param_list, args...);
	uint32_t mLookAhead = 20, mIterations = 0, mRandomMappings = 10000, hops = 1;
	bool enable_fidelity {false};
	uint32_t max_partial{(std::numeric_limits<uint32_t>::max)()}, max_children{(std::numeric_limits<uint32_t>::max)()};
	switch (param_list.size()) {
		case 4:
		if (std::holds_alternative<uint32_t>(param_list[3])) {
			hops			= std::get<uint32_t>(param_list[3]);
		}
		[[fallthrough]];
	case 3:
		if (std::holds_alternative<uint32_t>(param_list[2])) {
			max_children	= mRandomMappings = std::get<uint32_t>(param_list[2]);
		}
		[[fallthrough]];
	case 2:
		if (std::holds_alternative<uint32_t>(param_list[1])) {
			max_partial		= mIterations = std::get<uint32_t>(param_list[1]);
		}
		[[fallthrough]];
	case 1:
		if (std::holds_alternative<uint32_t>(param_list[0])) {
			mLookAhead		= std::get<uint32_t>(param_list[0]);
		}
		else if (std::holds_alternative<bool>(param_list[0])) {
			enable_fidelity	= std::get<bool>(param_list[0]);
		}
		break;
	default:
		break;
	}

	if (std::is_same_v<T, QPanda::SabreQAllocator>) {
		auto allocator = QPanda::SabreQAllocator::Create(graph, mLookAhead, mIterations, mRandomMappings);
		allocator->set_hops(hops);
		if (allocator) {
			return allocator->select_best_qubits_blocks(graph, prog);
		}
	}
	else if (std::is_same_v<T, QPanda::OptBMTQAllocator>) {
		auto allocator = QPanda::OptBMTQAllocator::Create(graph, enable_fidelity, max_partial, max_children);
		allocator->set_hops(hops);
		if (allocator) {
			return allocator->select_best_qubits_blocks(graph, prog);
		}
	}
	else {
        throw std::runtime_error("Unsupported QubitMapping type");
	}

	return {};
	//return allocator->select_best_qubits_blocks(graph, prog);
}