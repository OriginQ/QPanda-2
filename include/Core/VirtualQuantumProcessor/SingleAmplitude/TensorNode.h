#ifndef TENSOR_NODE_H
#define TENSOR_NODE_H

#include "Core/VirtualQuantumProcessor/SingleAmplitude/Tensor.h"
class Edge;
class Vertice;
class VerticeMatrix;
class QProgMap;

typedef std::vector<std::pair<qsize_t, qsize_t>> qubit_vector_t;
typedef std::map<size_t, Edge> edge_map_t;
typedef std::map<qsize_t, Vertice> vertice_map_t;
typedef std::vector<vertice_map_t> vertice_matrix_t;

typedef struct QubitVertice
{
    qsize_t m_qubit_id{SIZE_MAX};
    qsize_t m_num{SIZE_MAX};
    qsize_t m_max{SIZE_MAX};
    qsize_t m_count{SIZE_MAX};
}qubit_vertice_t;

class Edge
{
public:
    Edge(qsize_t qubit_count, ComplexTensor &tensor,
        std::vector<std::pair<qsize_t, qsize_t>> &contect_vertice);
    ~Edge() {};

    void earseContectVertice(qsize_t qubit, size_t num);
    qsize_t getQubitCount()const noexcept;
    bool mergeEdge(Edge & edge);

    void dimDecrementbyValue(qsize_t qubit,qsize_t num, int value);
    void dimDecrement(qsize_t qubit, qsize_t num);
    void dimIncrementByEdge(Edge &edge);
    void getEdgeMap(Edge &edge, size_t *mask);

    void mul(Edge& edge, size_t *mask_array);
    int getRank() const noexcept;
    ComplexTensor getComplexTensor() const noexcept;

    qcomplex_data_t getElem(VerticeMatrix &vertice);
    void setComplexTensor( ComplexTensor &tensor) noexcept;
    void getContectVertice(std::vector<std::pair<qsize_t, qsize_t>> & connect_vertice) const noexcept;
    void setContectVerticeVector(const qubit_vector_t &contect_vertice) noexcept;
    void setContectVertice(qsize_t qubit, qsize_t src_num, qsize_t des_num);

private:
    qsize_t m_qubit_count;
    ComplexTensor m_tensor;
    qubit_vector_t m_contect_vertice;
};

class Vertice
{
public:
    Vertice();
    Vertice(int value, std::vector<qsize_t> &contect_edge);
    ~Vertice();
    std::vector<qsize_t> & getContectEdge() noexcept;
    void addContectEdge(qsize_t) noexcept;

    void setContectEdge(qsize_t, qsize_t);
    void setContectEdgebyID(qsize_t id, qsize_t value);
    void deleteContectEdge(qsize_t);
    int getValue() const noexcept;

    void setValue(int) noexcept;
    void setNum(size_t num);
    size_t getNum();
private:
    std::vector<qsize_t> m_contect_edge;
    int m_value{-1};
    size_t m_num{0};
};

class VerticeMatrix
{
public:
    VerticeMatrix();
    qsize_t getQubitCount() const;

    qsize_t getVerticeCount() const;
    void subVerticeCount();
    qsize_t addVertice(qsize_t);
    qsize_t addVertice(qsize_t, qsize_t);

    qsize_t addVertice(qsize_t,qsize_t, Vertice &);
    int getVerticeValue(qsize_t, qsize_t);
    qubit_vertice_t  getVerticeByNum(size_t num);
    qsize_t getEmptyVertice();

    void setVerticeValue(qsize_t, qsize_t, int);
    void initVerticeMatrix(qsize_t);
    std::map<qsize_t, Vertice>::iterator deleteVertice(qsize_t, qsize_t);
    qsize_t getQubitVerticeLastID(qsize_t);

    std::vector<qsize_t> & getContectEdge(qsize_t, qsize_t);
    std::vector<qsize_t> & getContectEdgebynum(qsize_t, qsize_t);
    vertice_matrix_t::iterator begin()noexcept;
    vertice_matrix_t::iterator end()noexcept;

    vertice_matrix_t::iterator getQubitMapIter(qsize_t qubit)noexcept;
    vertice_map_t::iterator getQubitMapIterBegin(qsize_t qubit);
    vertice_map_t::iterator getQubitMapIterEnd(qsize_t qubit);
    vertice_map_t::iterator getVertice(qsize_t, qsize_t);

    void addContectEdge(qsize_t, qsize_t, qsize_t);
    void changeContectEdge(qsize_t, qsize_t, qsize_t,qsize_t);
    void deleteContectEdge(qsize_t, qsize_t, qsize_t);
    void clearVertice() noexcept;

    bool isEmpty() noexcept;
    void clear() noexcept;
    ~VerticeMatrix();
private:
    qsize_t m_qubit_count;
    qsize_t m_vertice_count;
    std::vector<std::map<qsize_t, Vertice>> m_vertice_matrix;
};

class QProgMap
{
public:
    QProgMap();
    ~QProgMap();
    size_t getMaxRank();
    size_t setMaxRank(size_t rank);

    void deleteMap();
    QProgMap(const QProgMap & old);
    QProgMap & operator = (const QProgMap & old);
    VerticeMatrix * getVerticeMatrix();

    size_t getQubitVerticeCount(qsize_t qubit_num);
    void setQubitNum(size_t num);
    bool isEmptyQProg();
    size_t getQubitNum();

    edge_map_t * getEdgeMap();
    void clearVerticeValue();
    void clear();
    qsize_t getVerticeCount() const;

    size_t m_count{0};
private:
    VerticeMatrix * m_vertice_matrix{nullptr};
    edge_map_t * m_edge_map{nullptr};
    size_t m_qubit_num{0};
    size_t m_max_tensor_rank{30};

};


#endif // !TENSOR_NODE_H
