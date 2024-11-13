#include "Core/Utilities/Tools/ArchGraph.h"
#include "Core//Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include <fstream>
#include <sstream>

using namespace QPanda;

ArchGraph::ArchGraph(uint32_t n, bool isGeneric)
    : WeightedGraph<double>(K_ARCH, n, Directed),
      mId(n, ""),
      mGeneric(isGeneric),
      mVID(0) {}

uint32_t ArchGraph::putVertex(std::string s) {
    if (mStrToId.find(s) != mStrToId.end())
        return mStrToId[s];

    uint32_t id = mVID++;
    mId[id] = s;
    mStrToId[s] = id;
    return id;
}

void ArchGraph::putReg(std::string id, std::string size) {
    mRegs.push_back(std::make_pair(id, std::stoul(size)));
}

std::vector<std::vector<int>> ArchGraph::get_adjacent_matrix()
{
	std::vector<std::vector<int>> adjacent_matrix(mN, std::vector<int>(mN, 0));
	for (uint32_t i = 0; i < mN; ++i)
	{
		auto adjacent = adj(i);

		for (uint32_t j : adjacent)
		{
			if (0 == adjacent_matrix[i][j])
			{
				adjacent_matrix[i][j] = 1;
				//adjacent_matrix[j][i] = 1;
			}
		}
	}

	return adjacent_matrix;
}

std::vector<std::pair<uint32_t, uint32_t>> ArchGraph::get_all_edges()
{
	std::vector<std::pair<uint32_t, uint32_t>> edges_vec;
	for (uint32_t i = 0; i < mN; ++i)
	{
		auto adjacent = adj(i);

		for (const uint32_t j : adjacent)
		{
			if (j < i){
				continue;
			}

			edges_vec.emplace_back(std::make_pair(i, j));
		}
	}

	return edges_vec;
}

std::vector<std::vector<double>> ArchGraph::get_adj_weight_matrix()
{
	std::vector<std::vector<double>> adj_weight_matrix(mN, std::vector<double>(mN, 0));
	for (uint32_t i = 0; i < mN; ++i)
	{
		std::set<uint32_t> adjacent;
		if (isDirectedGraph())
		{
			adjacent = c_succ(i);
		}
		else
		{
			adjacent = adj(i);
		}

		for (uint32_t j : adjacent)
		{
			if (1e-10 > abs(adj_weight_matrix[i][j]))
			{
				adj_weight_matrix[i][j] = getW(i, j);
				//adj_weight_matrix[j][i] = getW(j, i);
			}
		}
	}

	return adj_weight_matrix;
}

bool ArchGraph::isGeneric() {
    return mGeneric;
}

ArchGraph::RegsIterator ArchGraph::reg_begin() {
    return mRegs.begin();
}

ArchGraph::RegsIterator ArchGraph::reg_end() {
    return mRegs.end();
}

bool ArchGraph::ClassOf(const Graph* g) {
    return g->isArch();
}

std::unique_ptr<ArchGraph> ArchGraph::Create(uint32_t n) {
    return std::unique_ptr<ArchGraph>(new ArchGraph(n));
}

/*******************************************************************
*                 JsonFields
********************************************************************/
const std::string JsonFields<ArchGraph>::_quantum_chip_arch_label = "QuantumChipArch";
const std::string JsonFields<ArchGraph>::_qubits_label = "QubitCount";
const std::string JsonFields<ArchGraph>::_name_label = "name";
const std::string JsonFields<ArchGraph>::_adj_list_label = "adj";
const std::string JsonFields<ArchGraph>::_v_label = "v";
const std::string JsonFields<ArchGraph>::_weight_label = "w";

/*******************************************************************
*                 JsonBackendParser
********************************************************************/
std::unique_ptr<ArchGraph> JsonBackendParser<ArchGraph>::Parse(const rapidjson::Value& root) 
{
    auto &quantum_chip_arch = root[JsonFields<ArchGraph>::_quantum_chip_arch_label.c_str()];

	std::string name = "quantum_chip";
	if (quantum_chip_arch.HasMember(JsonFields<ArchGraph>::_name_label.c_str()))
	{
		name = quantum_chip_arch[JsonFields<ArchGraph>::_name_label.c_str()].GetString();
	}

	if (!quantum_chip_arch[JsonFields<ArchGraph>::_qubits_label.c_str()].IsUint())
	{
		QCERR_AND_THROW(run_fail, "Error: ArchGraph json error.");
	}
	const uint32_t qubits = quantum_chip_arch[JsonFields<ArchGraph>::_qubits_label.c_str()].GetUint();

	auto graph = ArchGraph::Create(qubits);
	graph->putReg(name, std::to_string(qubits));

	std::string adj_mat;
	if (quantum_chip_arch.HasMember(JsonFields<ArchGraph>::_adj_list_label.c_str()))
	{
		adj_mat = JsonFields<ArchGraph>::_adj_list_label;
	}
	else if (quantum_chip_arch.HasMember("AdjMatrix")) {
		adj_mat = "AdjMatrix";
	}
	
	if (adj_mat.empty())
	{
		QCERR_AND_THROW(run_fail, "Error: ArchGraph json error, no adjacent-matrix config.");
	}
    auto &adj = quantum_chip_arch[adj_mat.c_str()];

	if (qubits < adj.MemberCount())
	{
		QCERR_AND_THROW(run_fail, "Error: ArchGraph json cofigure error.");
	}

	uint32_t qubit_index = 0;
	for (rapidjson::Value::ConstMemberIterator iter = adj.MemberBegin(); iter != adj.MemberEnd(); ++iter, ++qubit_index)
	{
		std::string str_key = iter->name.GetString();
		uint32_t key = stoi(str_key);
#if 0
		if (key != qubit_index)
		{
			QCERR_AND_THROW(run_fail, "Error: ArchGraph json index error occurs.");
		}
#endif

		auto &iList = iter->value;
		if (!iList.IsArray())
		{
			QCERR_AND_THROW(run_fail, "Error: ArchGraph json error.");
		}

        for (uint32_t j = 0, f = iList.Size(); j < f; ++j) 
		{
            auto &jElem = iList[j];
			if (!jElem[JsonFields<ArchGraph>::_v_label.c_str()].IsInt())
			{
				QCERR_AND_THROW(run_fail, "Error: ArchGraph json error.");
			}

            auto v = jElem[JsonFields<ArchGraph>::_v_label.c_str()].GetUint();

			if (key == v) {
				QCERR_AND_THROW(run_fail, "Error: ArchGraph structure error, illegal spin edge.");
			}
            graph->putEdge(key, v, 1);//default

            if (jElem.HasMember(JsonFields<ArchGraph>::_weight_label.c_str()))
			{
				double w = 0;
				if (jElem[JsonFields<ArchGraph>::_weight_label.c_str()].IsDouble())
				{
					w = jElem[JsonFields<ArchGraph>::_weight_label.c_str()].GetDouble();
				}
				else if (jElem[JsonFields<ArchGraph>::_weight_label.c_str()].IsUint())
				{
					w = jElem[JsonFields<ArchGraph>::_weight_label.c_str()].GetUint();
				}
				else
				{
					QCERR_AND_THROW(run_fail, "Error: ArchGraph json error.");
				}

				if ((w < 0) || (1.0 < w )){
					QCERR_AND_THROW(init_fail, "Error: ArchGraph json config error,Double-gate fidelity configuration error.");
				}

                // In the Json, the standard is to have the probability of error.
                // What we want is the probability of succes.
                graph->setW(key, v, w);
            }
        }
    }

    return graph;
}