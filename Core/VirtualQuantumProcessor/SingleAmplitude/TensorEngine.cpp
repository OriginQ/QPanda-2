#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorEngine.h"
#include <algorithm>
#include "QPandaConfig.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

int log_2(int iNumber)
{
    if (iNumber <= 0)
    {
        abort();
    }

    for (int i = 0; i < 32; ++i)
    {
        if (0 == (iNumber >> i))
        {
            return i - 1;
        }
    }

    return -1;
}

static void mergeVerticeAndEdge(QuantumProgMap * prog_map,qsize_t qubit)
{
    auto vertice = prog_map->getVerticeMatrix();
    auto vertice_map_iter = vertice->getQubitMapIter(qubit);
    auto i = vertice->getQubitMapIterBegin(qubit);
    while (i != vertice->getQubitMapIterEnd(qubit))
    {
        auto contect_edge = vertice->getContectEdge(qubit, (*i).first);
        auto contect_count = contect_edge.size();
        auto edge_map = prog_map->getEdgeMap();

        if (2 == contect_count)
        {
            auto edge_iter_second = edge_map->find(contect_edge[1]);
            auto edge_second = (*edge_iter_second).second;
            auto edge_iter = edge_map->find(contect_edge[0]);
            auto edge_first = (*edge_iter).second;

            if ((edge_second.getRank() != 2) ||
                (edge_first.getRank() != 2))
            {
                ++i;
                continue;
            }
            if(((*edge_iter).second.getQubitCount() != 1) ||
                ((*edge_iter_second).second.getQubitCount() != 1))
            {
                ++i;
                continue;
            }
            (*edge_iter).second.premultiplication((*edge_iter_second).second);

            edge_map->erase(edge_iter_second);
            i = (*vertice_map_iter).erase(i);
            vertice->subVerticeCount();
            (*i).second.setContectEdgebyID(0, (*edge_iter).first);
        }
        else
        {
            ++i;
        }
    }
}

void getNoValueMaxRankVertice(QuantumProgMap * prog_map,
                              qubit_vertice_t * qubit_vertice)
{
    if (nullptr == prog_map)
    {
        throw exception();
    }
    auto vertice = prog_map->getVerticeMatrix();
    auto vertice_matrix_iter = 
                        vertice ->getQubitMapIter(qubit_vertice->m_qubit_id);
    auto edge = prog_map->getEdgeMap();
    size_t max = 0;
    size_t num = 0;
    qsize_t target_id = 0;
    for (auto aiter = (*vertice_matrix_iter).begin();
		aiter != (*vertice_matrix_iter).end();++aiter)
    {
        size_t temp = (*aiter).second.getContectEdge().size();
        bool is_true = true;
        if (max < temp)
            if ((*aiter).second.getValue() < 0)
            {
				max = temp;
				target_id = (*aiter).first;
            }
    }

    qubit_vertice->m_num = target_id;
    qubit_vertice->m_max = max;
}

qubit_vertice_t TensorEngine::getNoValueVertice(QuantumProgMap & prog_map,size_t find_edge)
{
    qubit_vertice_t qubit_vertice;
    qubit_vertice.m_max = 0;
    qubit_vertice.m_num = 0;
    qubit_vertice.m_qubit_id = 0;
    auto vertice = prog_map.getVerticeMatrix();
    int i = 0;
#if 0
    if (0 == find_edge)
    {
        for (auto iter = vertice->begin(); iter != vertice->end(); iter++)
        {
            for (auto map_iter = (*iter).begin(); map_iter != (*iter).end(); ++map_iter)
            {
                if (-1 == (*map_iter).second.getValue())
                {
                    qubit_vertice.m_num = (*map_iter).first;
                    qubit_vertice.m_qubit_id = i;
                    qubit_vertice.m_max = (*map_iter).second.getContectEdge().size();
                    return qubit_vertice;
                }
            }
            i++;
        }
    }
    else
    {
        auto edge_map = prog_map.getEdgeMap();
        auto edge = edge_map->find(find_edge);

        auto contect_vertice = (*edge).second.getContectVertice();


        for (auto aiter : contect_vertice)
        {
            auto contect_edge_vector = vertice->getContectEdge(aiter.first, aiter.second);
            auto contect_edge_size = contect_edge_vector.size();

            int max_rank = 0;
            for (size_t i = 0; i < contect_edge_size; i++)
            {
                if (contect_edge_vector[i] != find_edge)
                {
                    auto rank = (*edge_map->find(contect_edge_vector[i])).second.getRank();
                    if (max_rank < rank)
                        max_rank = rank;
                }
            }
            auto min_num = contect_edge_size;//max_rank > contect_edge_size ? max_rank : contect_edge_size;
            if (qubit_vertice.m_max > min_num)
            {
                qubit_vertice.m_max = min_num;
                qubit_vertice.m_qubit_id = aiter.first;
                qubit_vertice.m_num = aiter.second;
            }
        }

    }
#endif // 0
#if 1

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
#endif // 0

    return qubit_vertice;
}

qubit_vertice_t TensorEngine::getNoValueAndContectEdgeMaxVertice
                                (QuantumProgMap & prog_map)
{
    auto vertice = prog_map.getVerticeMatrix();

    vector<qubit_vertice_t> qubit_vertice;
    qubit_vertice.resize(vertice->getQubitCount());

    qsize_t i = 0;
    auto size = vertice->getQubitCount();
#pragma omp parallel for
    for (long long i = 0; i < size; i++)
    {
        qubit_vertice[i].m_qubit_id = i;
        getNoValueMaxRankVertice(&prog_map, &qubit_vertice[i]);
    }

    qubit_vertice_t temp;
    temp.m_qubit_id = 0;
    temp.m_max = 0;
    temp.m_num = 0;
    for (auto aiter = qubit_vertice.begin();
         aiter != qubit_vertice.end();
         ++aiter)
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

#include <iostream>
void split(QuantumProgMap * prog_map,
                             qubit_vertice_t * qubit_vertice,
                             qcomplex_data_t * result)
{
    qubit_vertice_t  temp;
    if ((nullptr == prog_map) || (nullptr == result))
    {
        throw exception();
    }
    if (nullptr == qubit_vertice)
    {

        temp = TensorEngine::getNoValueAndContectEdgeMaxVertice(*prog_map);
        temp.m_count = 0;
        split(prog_map, &temp, result);
    }
    else
    {
        if (qubit_vertice->m_max < 9)
        {
            (*result) = TensorEngine::Merge(*prog_map, nullptr);
        }
        else
        {

            QuantumProgMap *new_map = new QuantumProgMap(*prog_map);

            TensorEngine::dimDecrementbyValue(*prog_map, *qubit_vertice, 0);
            temp = TensorEngine::getNoValueAndContectEdgeMaxVertice(*prog_map);
            temp.m_count = ++qubit_vertice->m_count;
            qcomplex_data_t result_zero(0);
            if (temp.m_count > 0)
            {
                split(prog_map, &temp, &result_zero);

                TensorEngine::dimDecrementbyValue(*new_map, *qubit_vertice, 1);
                qcomplex_data_t result_one(0);
                split(new_map, &temp, &result_one);
                delete new_map;
                *result = result_one + result_zero;
            }
            else
            {
                std::thread thread = std::thread(split, prog_map, &temp, &result_zero);

                TensorEngine::dimDecrementbyValue(*new_map, *qubit_vertice, 1);
                qcomplex_data_t result_one(0);
                split(new_map, &temp, &result_one);
                thread.join();
                delete new_map;
                *result = result_one + result_zero;
            }
        }
    }
}

qcomplex_data_t TensorEngine:: Merge(QuantumProgMap & prog_map,
                                    qubit_vertice_t *qubit_vertice)
{
    auto vertice = prog_map.getVerticeMatrix();
    size_t i = 0;
    for (auto iter = vertice->begin(); iter != vertice->end(); iter++)
    {
        for (auto map_iter = (*iter).begin(); map_iter != (*iter).end();)
        {
            qubit_vertice_t qubit_vertice_1;
            qubit_vertice_1.m_qubit_id = i;
            qubit_vertice_1.m_num = (*map_iter).first;
            map_iter = MergeQuantumProgMap(prog_map,
            qubit_vertice_1);
            if (map_iter == (*iter).end())
            {
                break;
            }
        }
        i++;
    }

    qcomplex_data_t result(0);
    result = TensorEngine::computing(prog_map);
    return result;
}

qcomplex_data_t TensorEngine::computing(QuantumProgMap & prog_map)
{
    auto edge_map = prog_map.getEdgeMap();
    qcomplex_data_t result = 1;

    for (auto iter = edge_map->begin(); iter != edge_map->end(); ++iter)
    {
        result *= (*iter).second.getElem(*prog_map.getVerticeMatrix());
    }
    return result;
} 

#include <iostream>

void sort(EdgeMap * edge_map,vector<qsize_t> & vector)
{
    int max_rank = 0;
    auto size = vector.size();
    if (size <= 1)
    {
        return;
    }
    
    for (size_t i = 0; i < size - 1; i++)
    {
        for (size_t j = 0; j < size - i -1; j++)
        {
            auto first_edge = edge_map->find(vector[j]);
            auto second_edge = edge_map->find(vector[j+1]);

            if ((*first_edge).second.getRank() > (*second_edge).second.getRank())
            {
                std::swap(vector[j], vector[j + 1]);
            }
        }
    }

    return;
}

map<qsize_t, Vertice>::iterator TensorEngine::MergeQuantumProgMap(QuantumProgMap & prog_map,
                                          qubit_vertice_t & qubit_vertice)
{
    auto vertice = prog_map.getVerticeMatrix();
    auto edge_map = prog_map.getEdgeMap();

    auto contect_edge = vertice->getContectEdge(qubit_vertice.m_qubit_id,
                                                qubit_vertice.m_num);
    sort(edge_map, contect_edge);
    auto first_edge = edge_map->find(contect_edge[0]);
    try
    {
        qsize_t i = 0;
        auto edge_iter = edge_map->find((*contect_edge.begin()));
        vector<pair<qsize_t, qsize_t>> vertice_vector;
        
        
        for (size_t i = 1; i < contect_edge.size(); i++)
        {
            auto edge = edge_map->find(contect_edge[i]);
            (*first_edge).second.mergeEdge((*edge).second);
        }
        (*first_edge).second.dimDecrement(qubit_vertice.m_qubit_id,
                                          qubit_vertice.m_num);

        for (auto contect_edge_iter : contect_edge)
        {
            auto iter = edge_map->find(contect_edge_iter);
            auto contect_vertice = (*iter).second.getContectVertice();

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
       return vertice->deleteVertice(qubit_vertice.m_qubit_id, qubit_vertice.m_num);
    }
    catch (const calloc_fail&e)
    {
        throw e;
    }
    auto contect_vertice = (*first_edge).second.getContectVertice();
    
}

void TensorEngine::dimDecrementbyValue(QuantumProgMap & prog_map,
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




