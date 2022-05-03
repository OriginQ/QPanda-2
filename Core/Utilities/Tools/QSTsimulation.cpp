#include "Core/Utilities/Tools/QSTsimulation.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Core.h"
#include <bitset>


USING_QPANDA
using namespace std;

QSTSIM::QSTSIM()
{
}

QSTSIM::QSTSIM(QCircuit& cir)
{
    m_source_cir << cir;
}

QSTSIM::~QSTSIM()
{

}


std::vector<double> QPanda::QST_simulation(QCircuit& cir, QVec& qvec, size_t shots)
{
    auto encoding_circuit = [](QVec& qubits, vector<double>& prob)->QCircuit
    {
        QCircuit prob_cir = amplitude_encode(qubits, prob);
        return prob_cir;
    };

    auto Tomography_cir = [](QCircuit source_cir, QCircuit prob_cir, Qubit* control_bit)->QCircuit
    {
        QCircuit tomo_cir = QCircuit();
        tomo_cir << H(control_bit)
            << X(control_bit)
            << source_cir.control(control_bit)
            << X(control_bit)
            << prob_cir.control(control_bit)
            << H(control_bit);
        return tomo_cir;
    };

    QSTSIM qst_sim(cir);
    QCircuit source_cir = qst_sim.get_source_circuit();
    size_t bits_number = source_cir.get_used_qubits(qvec);

    int dim = 1 << bits_number;
    double measure_num = shots * dim * std::log(dim) / threshold / threshold;
    long long N = ceil(measure_num);
    auto machine = shared_ptr<CPUQVM>(new CPUQVM());
    machine->init();
    auto qubits = machine->qAllocMany(bits_number + 1);
    auto cbits = machine->cAllocMany(bits_number + 1);

    QVec main_qubits(qubits.begin(), qubits.end() - 1);
    std::vector<QPanda::ClassicalCondition> main_cbits(cbits.begin(), cbits.end() - 1);
    Qubit* control_bit = qubits[bits_number];

    std::vector<double> prob_result;
    std::vector<string> source_state;
    QProg source_prog = QProg(cir);
    source_prog << MeasureAll(qubits, cbits);
    auto source_results = machine->runWithConfiguration(source_prog, cbits, N);
    int item = 1 << bits_number;
    int source_item_flag;
    for (int i = 0; i < item; i++)
    {
        source_item_flag = 0;
        bitset<16> bs(i);
        string present_i_bin = bs.to_string().substr(bs.size() - bits_number, bs.size());
        for (auto& ele : source_results)
        {
            if (stoi(ele.first, nullptr, 2) == i)
            {
                source_state.push_back(ele.first);
                prob_result.push_back(sqrt(double(ele.second) / double(N)));
                source_item_flag = 1;
                break;
            }
        }
        if (source_item_flag == 0)
        {
            source_state.push_back(present_i_bin);
            prob_result.push_back(0);
        }
    }
    
    QCircuit prob_cir = encoding_circuit(main_qubits, prob_result);
    QCircuit tomo_cir = Tomography_cir(source_cir, prob_cir, control_bit);
    QProg all_prog = QProg(tomo_cir);
    all_prog << MeasureAll(qubits, cbits);
    auto results = machine->runWithConfiguration(all_prog, cbits, N);
    std::vector<int> sign;
    std::map<string, double> middle_result;
    int fined_flag;
    for (int i = 0; i < 1 << bits_number; i++)
    {
        fined_flag = 0;
        bitset<16> bs(i);
        string present_i_bin = bs.to_string().substr(bs.size() - bits_number - 1, bs.size());
        for (auto& ele : results)
        {
            if (ele.first == present_i_bin && ele.first.at(0) == '0')
            {
                middle_result.insert(make_pair(ele.first, ele.second));
                int temp_sign = ele.second > 0.4 * prob_result[i] * prob_result[i] * N ? 1 : -1;
                fined_flag = 1;
                break;
            }
        }
        if (fined_flag == 0)
        {
            middle_result.insert(make_pair(present_i_bin, 0));
        }
    }
    for (auto& data : middle_result)
    {
        int i = 0;
        int temp_sign = data.second > 0.4 * prob_result[i] * prob_result[i] * N ? 1 : -1;
        sign.push_back(temp_sign);
        i++;
    }
    vector<double> final_res;
    int real_point = 0;
    for (int i = 0; i < prob_result.size(); i++)
    {
        if (stoi(source_state[real_point], nullptr, 2) == i)
        {
            final_res.push_back(sign[i] * prob_result[i]);
            real_point++;
        }
        else
        {
            final_res.push_back(0);
        }
    }
    return final_res;
}
