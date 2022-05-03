#include "TensorEngine.h"

using namespace std;
USING_QPANDA

void sort(edge_map_t* edge_map, vector<qsize_t>& vector)
{
	int max_rank = 0;
	auto size = vector.size();
	if (size <= 1)
		return;

	for (size_t i = 0; i < size - 1; i++)
	{
		for (size_t j = 0; j < size - i - 1; j++)
		{
			auto first_edge = edge_map->find(vector[j]);
			auto second_edge = edge_map->find(vector[j + 1]);

			if ((*first_edge).second.getRank() > (*second_edge).second.getRank())
			{
				std::swap(vector[j], vector[j + 1]);
			}
		}
	}
}

void getNoValueMaxRankVertice(QProgMap * prog_map,
                              qubit_vertice_t * qubit_vertice)
{
    if (nullptr == prog_map)
    {
        throw exception();
    }
    auto vertice = prog_map->getVerticeMatrix();
    auto vertice_matrix_iter = 
                        vertice ->getQubitMapIter(qubit_vertice->m_qubit_id);

    size_t max = 0;
    qsize_t target_id = 0;
    for (auto aiter = (*vertice_matrix_iter).begin();
		aiter != (*vertice_matrix_iter).end();++aiter)
    {
        size_t temp = (*aiter).second.getContectEdge().size();
        if (max < temp)
        {
            if ((*aiter).second.getValue() < 0)
            {
                max = temp;
                target_id = (*aiter).first;
            }
        }
    }

    qubit_vertice->m_num = target_id;
    qubit_vertice->m_max = max;
}

qubit_vertice_t TensorEngine::getNoValueVertice(QProgMap & prog_map,size_t find_edge)
{
    qubit_vertice_t qubit_vertice;
    qubit_vertice.m_max = 0;
    qubit_vertice.m_num = 0;
    qubit_vertice.m_qubit_id = 0;
    auto vertice = prog_map.getVerticeMatrix();
    int i = 0;

    for (auto iter = vertice->begin(); iter != vertice->end(); iter++)
    {
        for (auto map_iter = (*iter).begin(); map_iter != (*iter).end(); ++map_iter)
        {
            qubit_vertice.m_num = (*map_iter).first;
            qubit_vertice.m_qubit_id = i;
            qubit_vertice.m_max = (*map_iter).second.getContectEdge().size();
            return qubit_vertice;
        }
        i++;
    }

    return qubit_vertice;
}

qubit_vertice_t TensorEngine::getNoValueAndContectEdgeMaxVertice
                                (QProgMap & prog_map)
{
    auto vertice = prog_map.getVerticeMatrix();
    vector<qubit_vertice_t> qubit_vertice;
    qubit_vertice.resize(vertice->getQubitCount());
    auto size = vertice->getQubitCount();

#pragma omp parallel for
    for (long long i = 0; i < size; i++)
    {
        qubit_vertice[i].m_qubit_id = i;
        getNoValueMaxRankVertice(&prog_map, &qubit_vertice[i]);
    }

    qubit_vertice_t temp;
    temp.m_qubit_id = 0;
    temp.m_num = 0;

    for (auto aiter = qubit_vertice.begin();
         aiter != qubit_vertice.end(); ++aiter)
    {
        if ((*aiter).m_max > temp.m_max)
        {
            temp.m_num = (*aiter).m_num;
            temp.m_qubit_id = (*aiter).m_qubit_id;
            temp.m_max = (*aiter).m_max;
        }
    }
    return temp;
} 


void TensorEngine::split(QProgMap &prog_map, qubit_vertice_t &qubit_vertice)
{
    qubit_vertice_t  temp;
	if (prog_map.m_count < 0)
	{
		if (SIZE_MAX == qubit_vertice.m_qubit_id || SIZE_MAX == qubit_vertice.m_num)
		{
            return;
		}

        TensorEngine::dimDecrementbyValue(prog_map, qubit_vertice, 0);
		prog_map.m_count++;
		auto max_rank = TensorEngine::getMaxRank(prog_map);
		if (max_rank < 25)
		{
            return;
		}
		temp = TensorEngine::getMaxQubitVertice(prog_map);
        TensorEngine::split(prog_map, temp);
	}
	else
	{
		throw std::runtime_error("both memory and computer node is not enough!");
	}
}

qcomplex_data_t TensorEngine::Merge(QProgMap &prog_map,
                                     const qprog_sequence_t &sequence)
{
    for (auto iter = sequence.begin(); iter != sequence.end(); ++iter)
    {
        auto vertice = prog_map.getVerticeMatrix()->getVerticeByNum(iter->first);

        if (vertice.m_qubit_id == SIZE_MAX || vertice.m_num == SIZE_MAX)
            continue;

        bool is_success = false;
        if (iter->second)
        {
            auto qubit_vertice_max = TensorEngine::getMaxQubitVertice(prog_map);
            if (qubit_vertice_max.m_qubit_id == SIZE_MAX ||
                qubit_vertice_max.m_num == SIZE_MAX)
            {
                continue;
            }

            TensorEngine::split(prog_map, qubit_vertice_max);
        }
        else
        {
            TensorEngine::MergeQuantumProgMap(prog_map, vertice, is_success);
            if (!is_success)
            {
                throw std::runtime_error("Real MergeQuantumProgMap error");
            }
        }
    }
    return TensorEngine::computing(prog_map);
}


qcomplex_data_t TensorEngine::computing(QProgMap & prog_map)
{
    auto edge_map = prog_map.getEdgeMap();
    qcomplex_data_t result = 1;
    for (auto iter = edge_map->begin(); iter != edge_map->end(); ++iter)
    {
        result *= (*iter).second.getElem(*prog_map.getVerticeMatrix());
    }
    return result;
} 


std::map<qsize_t, Vertice>::iterator TensorEngine::MergeQuantumProgMap(QProgMap & prog_map,
                                          qubit_vertice_t & qubit_vertice, bool &is_success)
{
    auto vertice = prog_map.getVerticeMatrix();
    auto edge_map = prog_map.getEdgeMap();

    auto contect_edge = vertice->getContectEdge(qubit_vertice.m_qubit_id,
                                                qubit_vertice.m_num);
    sort(edge_map, contect_edge);
    auto first_edge = edge_map->find(contect_edge[0]);

    qsize_t i = 0;
    auto edge_iter = edge_map->find((*contect_edge.begin()));
    vector<pair<qsize_t, qsize_t>> vertice_vector;

    for (size_t i = 1; i < contect_edge.size(); i++)
    {
        auto edge = edge_map->find(contect_edge[i]);
        if(edge!= edge_map->end())
            (*first_edge).second.mergeEdge(edge->second);
    }

    size_t memory_use = 0;
    int test_count = 0;
    for (auto i = edge_map->begin(); i != edge_map->end(); ++i)
    {
        test_count++;
        int rank = i->second.getRank()+3;
        memory_use+= 1ull << rank;
    }

    if(memory_use >= (1ull<<(prog_map.getMaxRank()+3)))
    {
        is_success = false;
        return map<qsize_t, Vertice>::iterator();
    }

    (*first_edge).second.dimDecrement(qubit_vertice.m_qubit_id,
                                      qubit_vertice.m_num);

    for (auto contect_edge_iter : contect_edge)
    {
        auto iter = edge_map->find(contect_edge_iter);
        vector<pair<qsize_t, qsize_t>> contect_vertice;
        (*iter).second.getContectVertice(contect_vertice);

        for (auto contect_vertice_iter : contect_vertice)
        {
            if ((contect_vertice_iter.first != qubit_vertice.m_qubit_id) ||
                    (contect_vertice_iter.second != qubit_vertice.m_num))
            {
                vertice->deleteContectEdge(contect_vertice_iter.first,
                                           contect_vertice_iter.second,
                                           contect_edge_iter);
                vertice->addContectEdge(contect_vertice_iter.first,
                                        contect_vertice_iter.second,
                                        (*edge_iter).first);
            }
        }
        if (0 != i)
        {
            edge_map->erase(iter);
        }
        i++;
    }
    is_success = true;
    return vertice->deleteVertice(qubit_vertice.m_qubit_id, qubit_vertice.m_num);
}


void TensorEngine::MergeByVerticeVector(QProgMap & prog_map ,
                                        qprog_sequence_t &sequence)
{
    for (auto iter = sequence.begin(); iter != sequence.end(); ++iter)
    {
        auto vertice = prog_map.getVerticeMatrix()->getVerticeByNum(iter->first);

        if (vertice.m_qubit_id == SIZE_MAX || vertice.m_num == SIZE_MAX)
            continue;

        bool is_success = false;
        if (iter->second)
        {
            auto qubit_vertice_max = TensorEngine::getMaxQubitVertice(prog_map);
            if (qubit_vertice_max.m_qubit_id == SIZE_MAX || qubit_vertice_max.m_num == SIZE_MAX)
            {
                continue;
            }

            TensorEngine::split(prog_map, qubit_vertice_max);
        }
        else
        {
            TensorEngine::MergeQuantumProgMap(prog_map, vertice, is_success);
            if (!is_success)
            {
                throw std::runtime_error("Real MergeQuantumProgMap error");
            }
        }
    }
}


void TensorEngine::seq_merge_by_vertices(QProgMap& prog_map,
	std::vector<size_t> vertice_vector,
	qprog_sequence_t& sequence)
{
	QProgMap* bak_map = nullptr;
	for (auto iter = vertice_vector.begin(); iter != vertice_vector.end(); ++iter)
	{
		auto vertice = prog_map.getVerticeMatrix()->getVerticeByNum(*iter);
		if (vertice.m_qubit_id == SIZE_MAX || vertice.m_num == SIZE_MAX)
			continue;

		bool is_success = false;
		bak_map = new QProgMap(prog_map);
		TensorEngine::MergeQuantumProgMap(prog_map, vertice, is_success);

		if (is_success)
		{
			delete bak_map;
			bak_map = nullptr;
			sequence.push_back({ *iter, false });
		}
		else
		{
			prog_map = *bak_map;
			auto qubit_vertice_max = TensorEngine::getMaxQubitVertice(prog_map);

			if (qubit_vertice_max.m_qubit_id == SIZE_MAX || qubit_vertice_max.m_num == SIZE_MAX)
			{
				continue;
			}

			sequence.push_back({ *iter, true });
            TensorEngine::split(prog_map, qubit_vertice_max);
			--iter;
		}
	}

	return;
}

void TensorEngine::seq_merge(QProgMap& prog_map, qprog_sequence_t& vertice_vector)
{
	auto vertice = prog_map.getVerticeMatrix();
	QProgMap* bak_map = nullptr;
	QubitVertice qubit_vertice;

	size_t i = 0;
	bool flag = false;
	auto iter_row = vertice->begin();

	while (iter_row != vertice->end())
	{
		auto iter_col = (*iter_row).begin();
		while (iter_col != (*iter_row).end())
		{
			qubit_vertice.m_qubit_id = i;
			qubit_vertice.m_num = (*iter_col).first;
			size_t vertice_num = iter_col->second.getNum();
			bak_map = new QProgMap(prog_map);

			bool is_success = false;
			auto tmp_map_iter = TensorEngine::MergeQuantumProgMap(prog_map, qubit_vertice, is_success);
			if (is_success)
			{
				delete bak_map;
				bak_map = nullptr;
				vertice_vector.push_back({ vertice_num, false });
				iter_col = tmp_map_iter;
			}
			else
			{
				prog_map = *bak_map;
				auto qubit_vertice_max = TensorEngine::getMaxQubitVertice(prog_map);
				vertice_vector.push_back({ vertice_num, true });

				TensorEngine::split(prog_map, qubit_vertice_max);
				vertice = prog_map.getVerticeMatrix();
				iter_row = vertice->begin();
				flag = true;
				i = 0;
				break;
			}
		}
		if (flag)
		{
			flag = false;
			continue;
		}
		i++;
		iter_row++;
	}
}

void TensorEngine::dimDecrementbyValue(QProgMap & prog_map,
    qubit_vertice_t & qubit_vertice,int value)
{
    auto vertice = prog_map.getVerticeMatrix();
    auto edge_map = prog_map.getEdgeMap();
    auto contect_edge = vertice->getContectEdge(qubit_vertice.m_qubit_id,qubit_vertice.m_num);

    for (auto iter : contect_edge)
    {
        auto find_iter =edge_map->find(iter);
        if (find_iter != edge_map->end())
        {
            (*find_iter).second.dimDecrementbyValue(qubit_vertice.m_qubit_id,
                                                    qubit_vertice.m_num,
                                                    value);
        }
    }

    vertice->deleteVertice(qubit_vertice.m_qubit_id,
                           qubit_vertice.m_num);
}

void TensorEngine::dimDecrementbyValueAndNum(QProgMap & prog_map,
    qubit_vertice_t & qubit_vertice,int value)
{
    auto vertice = prog_map.getVerticeMatrix();
    auto edge_map = prog_map.getEdgeMap();
    auto contect_edge = vertice->getContectEdgebynum(qubit_vertice.m_qubit_id,qubit_vertice.m_num);

    for (auto iter : contect_edge)
    {
        auto find_iter =edge_map->find(iter);
        if (find_iter != edge_map->end())
        {
            (*find_iter).second.dimDecrementbyValue(qubit_vertice.m_qubit_id,
                                                    qubit_vertice.m_num,
                                                    value);
        }
    }

    vertice->deleteVertice(qubit_vertice.m_qubit_id,
                           qubit_vertice.m_num);
}

size_t TensorEngine::getMaxRank(QProgMap &prog_map)
{
    auto edge_map = prog_map.getEdgeMap();
    size_t max_rank = 0;

    for (auto &edge : (*edge_map))
    {
        size_t tmp_rank = edge.second.getRank();
        if (tmp_rank > max_rank)
        {
            max_rank = tmp_rank;
        }
    }

    return max_rank;
}

qubit_vertice_t TensorEngine::getMaxQubitVertice(QProgMap &prog_map)
{
    auto vertice = prog_map.getVerticeMatrix();
    auto edge_map = prog_map.getEdgeMap();

    auto max_rank = 0;

    auto max_rank_edge_num = 0;
    for (auto &edge : (*edge_map))
    {
        auto rank = edge.second.getRank();
        if (rank > max_rank)
        {
            max_rank = rank;

            max_rank_edge_num = edge.first;

        }
    }

    qubit_vector_t max_edge_connect;

    auto iter = edge_map->find(max_rank_edge_num);
    if(iter == edge_map->end())
    {
        QCERR("error");
        throw runtime_error("error");
    }
    iter->second.getContectVertice(max_edge_connect);
    qsize_t vertice_num = 0;
    qubit_vertice_t qubit_vertice_max;
    for (auto &val : max_edge_connect)
    {
        auto qubit_vertice =vertice->getVertice(val.first,val.second);
        auto connect_edge_count = qubit_vertice->second.getContectEdge().size();
        if(vertice_num <= connect_edge_count)
        {
            vertice_num = connect_edge_count;
            qubit_vertice_max.m_qubit_id = val.first;
            qubit_vertice_max.m_num = val.second;
        }
    }

    return qubit_vertice_max;
}

void TensorEngine::getVerticeMap(QProgMap & prog_map, vector<pair<size_t, size_t>> & map_vector)
{
    auto vertice_matrix = prog_map.getVerticeMatrix();
    auto qubit_count = prog_map.getQubitNum();
    auto edge_map = prog_map.getEdgeMap();
    for (size_t i = 0; i < qubit_count; i++)
    {
        for (auto iter = vertice_matrix->getQubitMapIterBegin(i);
            iter != vertice_matrix->getQubitMapIterEnd(i); ++iter)
        {
            auto connect_edge = iter->second.getContectEdge();
            for (size_t j = 0; j < connect_edge.size(); j++)
            {
                auto edge = edge_map->find(connect_edge[j]);
                qubit_vector_t edge_connect_vertice_vector;
                edge->second.getContectVertice(edge_connect_vertice_vector);
                for (size_t k = 0; k < edge_connect_vertice_vector.size(); k++)
                {
                    pair<size_t, size_t> temp;
                    temp.first = iter->second.getNum();
                    auto edge_connect_vertice = vertice_matrix->getVertice(edge_connect_vertice_vector[k].first,
                        edge_connect_vertice_vector[k].second);
                    temp.second = edge_connect_vertice->second.getNum();

                    if (temp.first == temp.second)
                        continue;
                    map_vector.push_back(temp);
                }
            }
        }
    }
}

