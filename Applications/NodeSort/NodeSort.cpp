#include <string.h>
#include <iostream>
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
#include "Core/Utilities/Tools/QStatMatrix.h"

#include "Extensions/Extensions.h"


using namespace std;
USING_QPANDA

vector<vector<double>> genGraph(const string& grid, int& real_size);
int calcNodeSort(
    const vector<vector<double>>& graph,
    int real_size,
    float lambda,
    float lambda1, 
    float lambda2,
    const std::string &filename);
int calcSIR(
    const vector<vector<double>>& graph, 
    int real_size,
    std::vector<int> sources, 
    float infectious_rate, 
    float recovery_rate, 
    int timestep,
    const std::string& filename);

int main(int argc, char* argv[])
{
    string grid = "1-2";
    float lambda = 0.2;
    float lambda1 = 0.5;
    float lambda2 = 0.5;
    bool sir = false;
    std::vector<int> sources;
    float infectious_rate = 0.9;
    float recovery_rate = 0.1;
    int timestep = 5;
    string filename = "result.json";

    if (argc == 2 && !strcmp(argv[1], "--help")) {
        cout << "--grid \"0-1;0-2\" (necessary. 0-1;0-2 represent the grid is 1-0-2)" << endl;
        cout << "--lambda 0.2 (default = 0.2)" << endl;
        cout << "--lambda1 0.5 (default = 0.5)" << endl;
        cout << "--lambda2 0.5 (default = 0.5)" << endl;
        cout << "--SIR" << endl;
        cout << "--source \"0,1\" (default = 0)" << endl;
        cout << "--infectious_rate 0.9 (default = 0.9)" << endl;
        cout << "--recovery_rate 0.1 (default = 0.1)" << endl;
        cout << "--timestep 5 ([5~10]default = 5)" << endl;
        cout << "--filename result.json (default = result.json)" << endl;

        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }
    
    for (int i = 1; i < argc;) {
        if (!strcmp(argv[i], "--grid")) {
            grid = argv[i + 1];
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--lambda")) {
            lambda = atof(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--lambda1")) {
            lambda1 = atof(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--lambda2")) {
            lambda2 = atof(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--SIR")) {
            sir = true;
            i = i + 1;
        }
        else if (!strcmp(argv[i], "--source")) {
            auto str_source = QString(argv[i + 1]);
            auto items = str_source.split(",", QString::SkipEmptyParts);
            for (auto& i : items)
            {
                bool ok = false;
                auto source = i.toInt(&ok);
                if (!ok)
                {
                    cout << "source config error!" << endl;
                    return -1;
                }
                sources.push_back(source);
            }
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--infectious_rate")) {
            infectious_rate = atof(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--recovery_rate")) {
            recovery_rate = atof(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--timestep")) {
            timestep = atoi(argv[i + 1]);
            i = i + 2;
        }
        else if (!strcmp(argv[i], "--filename")) {
            filename = argv[i + 1];
            i = i + 2;
        }
    }

    if (sources.empty())
    {
        sources.push_back(0);
    }

    int real_size = 0;
    auto graph = genGraph(grid, real_size);
    if (graph.empty())
    {
        return -1;
    }

    if (sir)
    {
        return calcSIR(
            graph, 
            real_size, 
            sources, 
            infectious_rate, 
            recovery_rate, 
            timestep,
            filename);
    }
    else
    {
        return calcNodeSort(graph, real_size, lambda, lambda1, lambda2, filename);
    }

}

vector<vector<double>> genGraph(const string& grid, int &real_size)
{
    vector<vector<double>> graph;
    QString str_grid(grid);
    auto strs = str_grid.split(";", QString::SkipEmptyParts);

    real_size = 0;
    map<int, vector<int>> node_neighbors;
    for (auto& item : strs)
    {
        auto items = item.split("-", QString::SkipEmptyParts);
        if (items.size() > 2)
        {
            cerr << "grid format error! error item: " << item << endl;
            return graph;
        }
        
        bool ok = false;
        auto node = items[0].toInt(&ok);
        if (!ok)
        {
            cerr << "node must be digital! error node: " << item[0] << endl;
            return graph;
        }
        if (node > real_size)
        {
            real_size = node;
        }
        if (node_neighbors.find(node) == node_neighbors.end())
        {
            node_neighbors.insert(make_pair(node, vector<int>()));
        }
        
        if (items.size() == 2)
        {
            auto neighbor = items[1].toInt(&ok);
            if (!ok)
            {
                cerr << "node must be digital! error node: " << item[1] << endl;
                return graph;
            }

            if (neighbor > real_size)
            {
                real_size = neighbor;
            }
            node_neighbors[node].push_back(neighbor);
        }
    }
    auto size = node_neighbors.size();
    if (size == 0)
    {
        cerr << "no node found! error grid: " << grid << endl;
        return graph;
    }
    real_size++;

    int tmp_log = ceil(std::log2(real_size));
    int tmp_size = std::pow(2, tmp_log);
    graph.resize(tmp_size);
    for (auto i = 0; i < tmp_size; i++)
    {
        graph[i].resize(tmp_size);
        memset(graph[i].data(), 0, sizeof(int) * tmp_size);
    }

    for (auto& iter : node_neighbors)
    {
        if (iter.first >= real_size)
        {
            cerr << "node is not continus from 0. error node: " << iter.first << endl;
            return vector<vector<double>>();
        }
        for (auto &neighbor: iter.second)
        {
            if (neighbor >= real_size)
            {
                cerr << "node is not continus from 0. error node: " << neighbor << endl;
                return vector<vector<double>>();
            }

            graph[iter.first][neighbor] = 1.0;
            graph[neighbor][iter.first] = 1.0;
        }
    }

    return graph;
}

template<class T>
bool lessCmp(std::pair<int, T> p1, std::pair<int, T> p2)
{
    return p1.second > p2.second;
}

template<class T>
std::vector<std::pair<int, T>> quickSort(const std::vector<T>& vec)
{
    std::vector<std::pair<int, T>> sort_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        sort_vec.push_back(std::make_pair(i, vec[i]));
    }

    std::sort(sort_vec.begin(), sort_vec.end(), lessCmp<T>);

    return sort_vec;
}

int calcNodeSort(
    const vector<vector<double>>& graph,
    int real_size,
    float lambda,
    float lambda1,
    float lambda2,
    const std::string& filename)
{
    NodeSortProblemGenerator gen;
    gen.setProblemGraph(graph);
    gen.setLambda1(lambda);
    gen.setLambda2(lambda1);
    gen.setLambda3(lambda2);
    gen.exec();

    std::cout << "Classical Liner result: " << std::endl;
    auto c_result = gen.getLinearSolverResult();
    std::vector<double> tmp_result;
    for (auto i = 0; i < c_result.size(); i++)
    {
        tmp_result.push_back(c_result[i]);
    }
    auto c_sort_result = quickSort(tmp_result);
    for (int i = 0; i < c_sort_result.size(); i++)
    {
        cout << c_sort_result[i].first << ", " << c_sort_result[i].second << endl;
    }

    //std::cout << gen.getLinearSolverResult();
    std::cout << std::endl;

    auto oA = gen.getMatrixA();
    auto ob = gen.getVectorB();

    auto A = Eigen_to_QStat(oA);
    std::vector<double> b;
    for (auto i = 0; i < ob.size(); i++)
    {
        b.push_back(ob[i]);
    }

    std::cout << "HHL:" << std::endl;
    QStat result = HHL_solve_linear_equations(A, b);

    std::vector<double> result_vec;
    result_vec.resize(real_size);
    for (int i = 0; i < real_size; i++)
    {
        result_vec[i] = result[i].real();
        std::cout << result_vec[i] << std::endl;
    }
    
    auto sort_result = quickSort(result_vec);

    OriginCollection collection(filename, false);
    collection = { "sort_index", "value" };
    for (int i = 0; i < sort_result.size(); i++)
    {
        cout << sort_result[i].first << ", " << sort_result[i].second << endl;
        collection.insertValue(sort_result[i].first, sort_result[i].second);
    }

    if (!collection.write())
    {
        cerr << "write sort result failed!" << endl;
        return -1;
    }

    return 0;
}

int calcSIR(
    const vector<vector<double>>& graph,
    int real_size,
    std::vector<int> sources,
    float infectious_rate,
    float recovery_rate,
    int timestep,
    const std::string& filename)
{
    srand((int)time(0));

    vector<int> state;
    state.resize(real_size);
    memset(state.data(), 0, sizeof(int) * real_size);
    for (auto &i : sources)
    {
        if (i >= real_size)
        {
            cerr << "SIR source index error!" << endl;
            return -1;
        }
        state[i] = 1;
    }

    auto tmp_state = state;
    vector<vector<int>> time_states;
    for (int t = 0; t < timestep; t++) 
    {
        for (int i = 0; i < real_size; i++) 
        {
            if (state[i] == 1) {
                double r1 = 1.0 * rand() / RAND_MAX;
                if (r1 < recovery_rate) 
                {
                    tmp_state[i] = 2;
                }

                for (int j = 0; j < real_size; j++) {
                    double r2 = 1.0 * rand() / RAND_MAX;
                    if (graph[i][j] > 1e-3 
                        && r2 < infectious_rate
                        && state[j] == 0) 
                    {
                        tmp_state[j] = 1;
                    }
                }
            }
        }

        time_states.push_back(tmp_state);
        state = tmp_state;
    }

    OriginCollection collection(filename, false);
    collection = { "time_step", "node_state" };
    for (int i = 0; i < time_states.size(); i++)
    {
        collection.insertValue(i, time_states[i]);
    }

    if (!collection.write())
    {
        cerr << "write sort result failed!" << endl;
        return -1;
    }

    return 0;
}
