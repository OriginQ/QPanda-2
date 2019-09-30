#include "Components/MaxCutProblemGenerator/MaxCutProblemGenerator.h"
using namespace std;
double QPanda::vector_dot(vector<double> &x, vector<double> &y)
{
    if (x.size() != y.size())
    {
        QCERR("unmatched");
        throw std::runtime_error("unmatched");
    }
    double sum = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        sum += x[i] * y[i];
    }
    x.clear();
    y.clear();
    return sum;
}

double QPanda::all_cut_of_graph(vector<vector<double>> adjacent_matrix,
    vector<double> &all_cut_list,
    vector<size_t> &target_value_list)
{
    size_t dimension = adjacent_matrix.size();
    //py::dict max_sum = {};
    double sum = 0;
    double max_value = 0;
    target_value_list.clear();
    for (size_t i = 0; i < (1ull << dimension); ++i)
    {
        sum = 0;
        for (size_t j = 0; j < dimension; ++j)
        {
            for (size_t k = 0; k < dimension; ++k)
            {
                if ((i >> j) % 2 != (i >> k) % 2)
                {
                    sum += adjacent_matrix[j][k];
                }                
            }
        }
        all_cut_list[i] = sum;
        if (sum - max_value>1e-6)
        {
            target_value_list.clear();
            target_value_list.push_back(i);
            max_value = sum;
        }
        else if (abs(sum - max_value) < 1e-6)
        {
            target_value_list.push_back(i);
        }
    }
    return max_value;
}

