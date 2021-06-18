#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorNode.h"
#include <algorithm>
using namespace std;
USING_QPANDA

qsize_t Edge::getQubitCount() const noexcept
{
    return m_qubit_count;
}

bool Edge::mergeEdge(Edge & edge)
{
    dimIncrementByEdge(edge);
    size_t * mask_array = new size_t[edge.getRank()];
    if (mask_array == nullptr)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }
    edge.getEdgeMap(*this, mask_array);
    mul(edge, mask_array);
    delete[] mask_array;
    return true;
}

void Edge::dimDecrementbyValue(qsize_t qubit,qsize_t num, int value)
{
    if ((0 != value) && (1 != value))
    {
        throw exception();
    }

    try
    {
        int i = 1;
        for (auto iter = m_contect_vertice.begin();
            iter != m_contect_vertice.end();
            ++iter)
        {
            if (((*iter).first == qubit) &&
                ((*iter).second == num))
            {
                m_tensor.getSubTensor(i, value);
                iter = m_contect_vertice.erase(iter);
                return;
            }
            i++;
        }
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}


Edge::Edge(qsize_t qubit_count, ComplexTensor &tensor,
           vector<pair<qsize_t, qsize_t> > &contect_vertice)
     : m_qubit_count(qubit_count), m_tensor(tensor),
       m_contect_vertice(contect_vertice)
{

}

void Edge::earseContectVertice(qsize_t qubit, size_t num)
{
    int i = 1;
    for (auto iter = m_contect_vertice.begin();
        iter != m_contect_vertice.end();
        ++iter)
    {
        if (((*iter).first == qubit) && ((*iter).second == num))
        {
            iter = m_contect_vertice.erase(iter);
            return;
        }
        i++;
    }
}

void Edge::dimDecrement(qsize_t qubit, qsize_t num)
{
    try
    {
        int i = 1;
        for (auto iter = m_contect_vertice.begin();
            iter != m_contect_vertice.end();
            ++iter)
        {
            if (((*iter).first == qubit) && ((*iter).second == num))
            {
                m_tensor.dimDecrement(i);
                iter = m_contect_vertice.erase(iter);
                return;
            }
            i++;
        }
    }
    catch (const std::exception&e)
    {
        throw e;
    }

}

void Edge::dimIncrementByEdge(Edge & edge)
{
    size_t i = 0;
    
    for (auto iter : edge.m_contect_vertice)
    {
        bool is_true = false;
        for (auto this_iter : m_contect_vertice)
        {
            if ((iter.first == this_iter.first) &&
                (iter.second == this_iter.second))
            {
                is_true = true;
                break;
            }
        }
        if (!is_true)
        {
            m_contect_vertice.push_back(iter);
            i++;
        }

    }
    try
    {
        m_tensor.dimIncrement(i);
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}
void Edge::getEdgeMap(Edge &edge,
           size_t * mask)
{
    size_t i;
    size_t j = 0;
    vector<size_t> insert_vertice;
    for (auto &val : m_contect_vertice)
    {
        i = 0;
        bool is_true = false;
        for (auto &this_val : edge.m_contect_vertice)
        {
            if ((val.first == this_val.first) &&
                (val.second == this_val.second))
            {
                is_true = true;
                break;
            }
            i++;
        }

        if (is_true)
        {
            mask[j] = i;
        }
        j++;
    }
}

void Edge:: mul(Edge &edge,size_t * mask_array)
{
    m_tensor.mul(edge.m_tensor, mask_array);
}


int Edge::getRank() const noexcept
{
    return m_tensor.getRank();
}

qcomplex_data_t Edge::getElem(VerticeMatrix &vertice)
{
    qubit_vector_t vertice_vector;
    getContectVertice(vertice_vector);
    auto size = vertice_vector.size();
	if (m_tensor.getRank() == 0)
	{
		return m_tensor.getElem(0);
	}
    size_t M = 0;
    for (size_t i = 0; i < size; i++)
    {
        M += vertice.getVerticeValue(vertice_vector[i].first,
            vertice_vector[i].second) <<
            (size - i - 1);
    }
    auto result = m_tensor.getElem(M);
    return result;
}

void Edge::setComplexTensor( ComplexTensor &tensor) noexcept
{
    m_tensor = tensor;
}

void Edge::getContectVertice(vector<pair<qsize_t, qsize_t>> & connect_vertice) const noexcept
{
    connect_vertice.assign(m_contect_vertice.begin(),m_contect_vertice.end());
}

void Edge::setContectVerticeVector
           (const qubit_vector_t & contect_vertice) noexcept
{
    m_contect_vertice.resize(0);
    for (auto val : contect_vertice)
    {
        m_contect_vertice.push_back(val);
    }
}

void Edge::setContectVertice(qsize_t qubit, qsize_t src_num, qsize_t des_num)
{
    try
    {
        for (size_t i = 0; i < m_contect_vertice.size(); i++)
        {
            if ((m_contect_vertice[i].first == qubit) && 
                 (m_contect_vertice[i].second == src_num))
            {
                m_contect_vertice[i].second = des_num;
            }
        }
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}


Vertice::Vertice()
{

}

Vertice::Vertice(int value, vector<qsize_t> &contect_edge)
    :m_contect_edge(contect_edge), m_value(value), m_num(0)
{

}

Vertice::~Vertice()
{

}

vector<qsize_t> & Vertice::getContectEdge() noexcept
{
    return m_contect_edge;
}

void Vertice::addContectEdge(qsize_t edge_id) noexcept
{
    m_contect_edge.push_back(edge_id);
}

void Vertice::setContectEdge(qsize_t id, qsize_t value)
{
    try
    {
        for (auto i = m_contect_edge.begin();i != m_contect_edge.end(); ++i)
        {
            if (*i == id)
            {
                (*i) = value;
            }
        }
    }
    catch (const std::exception &e)
    {
        throw e;
    }

}

void Vertice::setContectEdgebyID(qsize_t id, qsize_t value)
{
    try
    {
        m_contect_edge[id] = value;
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

void Vertice::deleteContectEdge(qsize_t edge_num)
{
    for (auto iter = m_contect_edge.begin();
        iter != m_contect_edge.end();
        ++iter)
    {
        if (edge_num == *iter)
        {
            m_contect_edge.erase(iter);
            return;
        }
    }
}

int Vertice::getValue() const noexcept
{
    return m_value;
}

void Vertice::setValue(int value)noexcept
{
    m_value = value;
}

void Vertice::setNum(size_t num)
{
    m_num = num;
}

size_t Vertice::getNum()
{
    return m_num;
}

VerticeMatrix::VerticeMatrix()
    :m_qubit_count(0), m_vertice_count(0)
{

}

qsize_t VerticeMatrix::getQubitCount() const
{
    return m_qubit_count;
}

qsize_t VerticeMatrix::getVerticeCount() const
{
    return m_vertice_count;
}

void VerticeMatrix::subVerticeCount()
{
    if (m_vertice_count > 0)
    {
        m_vertice_count--;
    }
}

void VerticeMatrix::initVerticeMatrix(qsize_t qubit_num)
{
    m_qubit_count = qubit_num;
    m_vertice_count = qubit_num;
    for (qsize_t i = 0; i < qubit_num; i++)
    {
        Vertice temp;
        temp.setValue(0);
        temp.setNum(i);
        map<qsize_t, Vertice> vertice_vector;
        vertice_vector.insert(pair<qsize_t, Vertice>(0, temp));
        m_vertice_matrix.push_back(vertice_vector);
    }
}

map<qsize_t,Vertice>::iterator VerticeMatrix::deleteVertice(qsize_t qubit, 
                                                            qsize_t num)
{
    try
    {
        auto aiter = m_vertice_matrix[qubit].find(num);
        m_vertice_count--;
        return m_vertice_matrix[qubit].erase(aiter);
    }
    catch (const std::exception& e)
    {
        throw e;
    }

}


qsize_t VerticeMatrix::addVertice(qsize_t qubit)
{
    try
    {
        Vertice temp;
        qsize_t size = m_vertice_matrix[qubit].size();
        temp.setNum(m_vertice_count);
        m_vertice_matrix[qubit].insert(pair<qsize_t,Vertice>(size,temp));
        m_vertice_count++;
        return size;
    }
    catch (const std::exception& e)
    {
        throw e;
    }
}

qsize_t VerticeMatrix::addVertice(qsize_t qubit, qsize_t num)
{
    Vertice temp;
    return addVertice(qubit,num,temp);
}


qsize_t VerticeMatrix::addVertice(qsize_t qubit, qsize_t num,Vertice & vertice)
{
    try
    {
        auto is_true = m_vertice_matrix[qubit].
                       insert(pair<qsize_t, Vertice>(num, vertice));

        if (!is_true.second)
        {
            m_vertice_matrix[qubit][num] = vertice;
        }
        else
        {
            vertice.setNum(m_vertice_count);
            m_vertice_count++;
        }

        return num;
    }
    catch (const std::exception& e)
    {
        throw e;
    }
}

int VerticeMatrix::getVerticeValue(qsize_t qubit, qsize_t num)
{
    try
    {
        auto iter = m_vertice_matrix[qubit].find(num);
        return (*iter).second.getValue();
    }
    catch (const std::exception& e)
    {
        throw e;
    }
}

qubit_vertice_t VerticeMatrix::getVerticeByNum(size_t num)
{
    qubit_vertice_t temp;
    qsize_t qubit_num = 0;
    for (auto & i : m_vertice_matrix)
    {
        for (auto & j : i)
        {
            if (j.second.getNum() == num)
            {
                temp.m_qubit_id = qubit_num;
                temp.m_num = j.first;
                return temp;
            }
        }
        qubit_num++;
    }

    return temp;
}

qsize_t VerticeMatrix::getEmptyVertice()
{
    qsize_t result = 0;
    for (auto matrix_iter : m_vertice_matrix)
    {
        for (auto map_iter : matrix_iter)
        {
            auto vertice = map_iter.second.getContectEdge();
            if (0 == vertice.size())
            {
                result++;
            }
        }
    }
    return result;
}

void VerticeMatrix::setVerticeValue(qsize_t qubit, qsize_t num, int value)
{
    if (value > 1 && value < -1)
    {
        throw exception();
    }

    try
    {
        return m_vertice_matrix[qubit][num].setValue(value);
    }
    catch (const std::exception& e)
    {
        throw e;
    }
}

qsize_t VerticeMatrix::getQubitVerticeLastID(qsize_t qubit)
{
    try
    {
        auto iter = m_vertice_matrix[qubit].end();
        iter--;
        return (*iter).first;
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}

vector<qsize_t> & VerticeMatrix::getContectEdgebynum(qsize_t qubit,
                                               qsize_t num)
{
    try
    {
        if((qubit >= m_vertice_matrix.size()) ||(num > m_vertice_matrix[qubit].size()) )
        {
            QCERR("param error");
            throw run_fail("param error");
        }

        auto qubit_vertic_map = m_vertice_matrix[qubit];
        qsize_t i = 0;
        for (auto iter = qubit_vertic_map.begin() ;iter!= qubit_vertic_map.end();++iter) {
            if(i == num )
            {
                return (*iter).second.getContectEdge();
            }
            i++;
        }
    }
    catch (const std::exception&e)
    {
        throw e;
    }

}

vector<qsize_t> & VerticeMatrix::getContectEdge(qsize_t qubit, 
                                               qsize_t num) 
{
    try
    {
        auto iter = m_vertice_matrix[qubit].find(num);
        if(iter == m_vertice_matrix[qubit].end())
        {
            QCERR("iter is end");
            throw run_fail("iter is end");
        }
        return (*iter).second.getContectEdge();
    }
    catch (const std::exception&e)
    {
        throw e;
    }

}

vertice_matrix_t::iterator  VerticeMatrix::begin() noexcept
{
    return  m_vertice_matrix.begin();
}
vertice_matrix_t::iterator VerticeMatrix::end()noexcept
{
    return  m_vertice_matrix.end();
}

vertice_matrix_t::iterator VerticeMatrix::getQubitMapIter(qsize_t qubit) noexcept
{
    return m_vertice_matrix.begin() + qubit;
}

vertice_map_t::iterator VerticeMatrix::getQubitMapIterBegin(qsize_t qubit)
{
    try
    {
        return m_vertice_matrix[qubit].begin();
        
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

vertice_map_t::iterator VerticeMatrix::getQubitMapIterEnd(qsize_t qubit)
{
    try
    {
        return m_vertice_matrix[qubit].end();
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

vertice_map_t::iterator  VerticeMatrix::getVertice(qsize_t qubit, qsize_t num)
{
    try
    {
        return m_vertice_matrix[qubit].find(num);
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}

void VerticeMatrix::addContectEdge(qsize_t qubit, qsize_t vertice_num, qsize_t edge_id)
{
    try
    {
        auto contect_edge = m_vertice_matrix[qubit][vertice_num].getContectEdge();
        for (auto edge : contect_edge)
        {
            if (edge_id == edge)
            {
                return;
            }
        }
        m_vertice_matrix[qubit][vertice_num].addContectEdge(edge_id);
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}

void VerticeMatrix::changeContectEdge(qsize_t qubit,
                                      qsize_t vertice_num,
                                      qsize_t contect_edge_vector_num,
                                      qsize_t contect_edge)
{
    try
    {
        m_vertice_matrix[qubit][vertice_num].
            setContectEdge(contect_edge_vector_num,
                contect_edge);
    }
    catch (const std::exception&e)
    {
        throw e;
    }
}

void VerticeMatrix::deleteContectEdge(qsize_t qubit,
                                      qsize_t vertice_num,
                                      qsize_t contect_edge_vector_num )
{
    m_vertice_matrix[qubit][vertice_num].
        deleteContectEdge(contect_edge_vector_num);
}

void VerticeMatrix::clearVertice() noexcept
{
    for (auto iter = m_vertice_matrix.begin();
        iter != m_vertice_matrix.end();
        ++iter)
    {
        for (auto map_iter = (*iter).begin();
            map_iter != (*iter).end();
            ++map_iter)
        {
            (*map_iter).second.setValue(-1);
        }
    }
}

void VerticeMatrix::clear() noexcept
{
    m_vertice_matrix.clear();
    m_vertice_count = 0;
    m_qubit_count = 0;
}

bool VerticeMatrix::isEmpty() noexcept
{
    return m_vertice_matrix.empty() ||
           m_qubit_count == 0 ||
           m_vertice_count == 0;
}

VerticeMatrix::~VerticeMatrix()
{}

QProgMap::QProgMap()
{
    m_vertice_matrix = new VerticeMatrix();
    m_edge_map = new edge_map_t;
    m_max_tensor_rank = 30;
}

QProgMap::~QProgMap()
{
    deleteMap();
}

size_t QProgMap::getMaxRank()
{
    return m_max_tensor_rank;
}

size_t QProgMap::setMaxRank(size_t rank)
{
    m_max_tensor_rank = rank;
    return 0;
}

void QProgMap::deleteMap()
{
    if (nullptr != m_vertice_matrix)
    {
        delete m_vertice_matrix;
        m_vertice_matrix = nullptr;
    }

    if (nullptr != m_edge_map)
    {
        delete m_edge_map;
        m_edge_map = nullptr;
    }
}

QProgMap::QProgMap(const QProgMap &old)
{
    m_vertice_matrix = new VerticeMatrix(*(old.m_vertice_matrix));
    m_edge_map = new edge_map_t(*(old.m_edge_map));
    m_qubit_num = old.m_qubit_num;
    m_count = old.m_count;
    m_max_tensor_rank = old.m_max_tensor_rank;
}

QProgMap &QProgMap::operator =(const QProgMap &old)
{
    if (this == &old)
    {
        return *this;
    }

    if (nullptr == m_vertice_matrix)
    {
        delete  m_vertice_matrix;
    }

    if (nullptr == m_edge_map)
    {
        delete m_edge_map;
    }

    m_vertice_matrix = new VerticeMatrix(*(old.m_vertice_matrix));
    m_edge_map = new edge_map_t(*(old.m_edge_map));
    m_qubit_num = old.m_qubit_num;
    m_count = old.m_count;
    m_max_tensor_rank = old.m_max_tensor_rank;

    return *this;
}

VerticeMatrix *QProgMap::getVerticeMatrix()
{
    return m_vertice_matrix;
}

size_t QProgMap::getQubitVerticeCount(qsize_t qubit_num)
{
    if(m_vertice_matrix->getQubitCount()< qubit_num)
    {
        QCERR("qubit_num err");
        throw  std::invalid_argument("qubit_num err");
    }

    return m_vertice_matrix->getQubitMapIter(qubit_num)->size();
}

void QProgMap::setQubitNum(size_t num)
{
    m_qubit_num = num;
}

bool QProgMap::isEmptyQProg()
{
    return ((m_vertice_matrix->isEmpty()) ||
            (m_edge_map->empty()) ||
            (m_qubit_num == 0));
}

size_t QProgMap::getQubitNum()
{
    return m_qubit_num;
}

edge_map_t *QProgMap::getEdgeMap()
{
    return m_edge_map;
}

void QProgMap::clearVerticeValue()
{
    m_vertice_matrix->clearVertice();
}

void QProgMap::clear()
{
    m_vertice_matrix->clear();
    m_edge_map->clear();
    m_qubit_num = 0;
}

qsize_t QProgMap::getVerticeCount() const
{
    return m_vertice_matrix->getVerticeCount();
}

