#include "Core/Utilities/Tools/GraphDijkstra.h"
#include <queue>
using namespace std;
USING_QPANDA
GraphDijkstra::GraphDijkstra(const vector<vector<int> > &matrix) :
    m_vertex_count(0), m_edge(0)
{
    if (!matrix.size() && matrix.size() != matrix[0].size())
    {
        QCERR("Bad matrix");
        throw invalid_argument("Bad matrix");
    }

    m_vertex_count = (int)matrix.size();
    m_matrix = matrix;
    for (int i = 0; i < m_vertex_count; i++)
    {
        for (int j = 0; j < m_vertex_count; j++)
        {
            if (!m_matrix[i][j])
            {
                m_matrix[i][j] = kInfinite;
            }
        }
    }
    m_dist_vec.resize(m_vertex_count);
}

bool GraphDijkstra::dijkstra(int begin)
{
    if (begin < 1 || begin > m_vertex_count)
    {
        return false;
    }

    int i;
    for (i = 0; i < this->m_vertex_count; i++)
    {
        m_dist_vec[i].path_vec.push_back(begin);
        m_dist_vec[i].path_vec.push_back(i+1);
        m_dist_vec[i].value = m_matrix[begin-1][i];
    }

    m_dist_vec[begin - 1].value = 0;
    m_dist_vec[begin - 1].visit = true;

    int count = 1;
    while (count != m_vertex_count)
    {
        int temp = 0;
        int min = kInfinite;

        for (i = 0; i < this->m_vertex_count; i++)
        {
            if (!m_dist_vec[i].visit && m_dist_vec[i].value<min)
            {
                min = m_dist_vec[i].value;
                temp = i;
            }
        }

        m_dist_vec[temp].visit = true;
        count++;
        for (i = 0; i < this->m_vertex_count; i++)
        {
            if (!m_dist_vec[i].visit &&
                m_matrix[temp][i] != kInfinite &&
                (m_dist_vec[temp].value + m_matrix[temp][i]) < m_dist_vec[i].value)
            {
                m_dist_vec[i].value = m_dist_vec[temp].value + m_matrix[temp][i];
                m_dist_vec[i].path_vec = m_dist_vec[temp].path_vec;
                m_dist_vec[i].path_vec.push_back(i + 1);
            }
        }
    }
    return true;
}

int GraphDijkstra::getShortestPath(int begin, int end, vector<int> &path_vec)
{
    if (begin < 1 || end < 1 || begin > m_vertex_count || end > m_vertex_count)
    {
        return kError;
    }

    Dist dis;
    m_dist_vec.assign(m_vertex_count, dis);
    if(!dijkstra(begin))
    {
        return kError;
    }
    path_vec = m_dist_vec[end - 1].path_vec;
    return m_dist_vec[end - 1].value;
}

bool GraphDijkstra::is_connective()
{
    queue<int> temp_queue;
    vector<bool> visa_vertex_vec(m_vertex_count, false);
    int count=0;
    temp_queue.push(0);

    while(!temp_queue.empty())
    {
        int visa = temp_queue.front();
        visa_vertex_vec[visa] = true;
        temp_queue.pop();
        count++;

        for(int i = 0; i < m_vertex_count; i++)
        {
            if(kInfinite != m_matrix[visa][i] && !visa_vertex_vec[i])
            {
                temp_queue.push(i);
                visa_vertex_vec[i] = true;
            }
        }
    }
    if(count == m_vertex_count)
    {
        return true;
    }
    return false;
}

GraphDijkstra::~GraphDijkstra()
{ }
