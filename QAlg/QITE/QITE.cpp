#include "QAlg/QITE/QITE.h"
#include "Core/QuantumMachine/QuantumMachineFactory.h"
#include <Eigen/Eigenvalues>

QPANDA_BEGIN

QITE::QITE()
{
}

int QITE::exec()
{
    initEnvironment();
    srand((int)time(0));

    auto tmp_ansatz = m_ansatz;
    for (int t = 0; t < m_upthrow_num; t++)
    {
        for (int i = 0; i < m_iter_num; i++)
        {
            calcParaA();
            calcParaC();

            Eigen::MatrixXd A_inverse = pseudoinverse(m_A);
            Eigen::VectorXd theta_dot = -A_inverse * m_C;
            double delta_k = m_delta_tau * pow(m_Q, i);
            
            for (int j = 0; j < m_theta_index_vec.size(); j++)
            {
                auto t_j = m_theta_index_vec[j];
                double r = (rand() / double(RAND_MAX));
                if (m_update_mode == UpdateMode::GD_VALUE)
                {
                    m_ansatz[t_j].theta += theta_dot[j];
                }
                else if(m_update_mode == UpdateMode::GD_DIRECTION)
                {
                    if (theta_dot[j] > 0)
                    {
                        m_ansatz[t_j].theta -= delta_k * r;
                    }
                    else
                    {
                        m_ansatz[t_j].theta += delta_k * r;
                    }
                }
                else
                {
                    return -1;
                }

                int n_period = int(m_ansatz[t_j].theta / (2 * PI));
                if (m_ansatz[t_j].theta < 0)
                {
                    m_ansatz[t_j].theta += (n_period + 1) * 2 * PI;
                }
                else
                {
                    m_ansatz[t_j].theta -= n_period * 2 * PI;
                }
            }

            double tmp_exp = getExpectation(m_ansatz);
            m_log_writer << tmp_exp << std::endl;

            if (tmp_exp < m_expectation)
            {
                m_expectation = tmp_exp;
                m_best_ansatz = m_ansatz;
            }
        }
    }

    std::fstream para_out;
    para_out.open(m_log_file + "_para.txt", std::ios::out);
    para_out << "Last_para:" << std::endl;
    for (int i = 0; i < m_theta_index_vec.size(); i++)
    {
        para_out << m_ansatz[m_theta_index_vec[i]].theta << ", ";
    }

    para_out << std::endl << "Best_para:" << std::endl;
    for (int i = 0; i < m_theta_index_vec.size(); i++)
    {
        para_out << m_best_ansatz[m_theta_index_vec[i]].theta << ", ";
    }
    para_out.close();
    m_log_writer.close();

    m_ansatz = tmp_ansatz;
    //m_machine->Free_Qubits(m_qlist);

    return 0;
}

prob_tuple QITE::getResult()
{
    QProg prog;
    prog << constructCircuit(m_best_ansatz);
    m_machine->directlyRun(prog);

    auto temp = dynamic_cast<IdealMachineInterface*>(m_machine.get());
    if (nullptr == temp)
    {
        QCERR("m_machine is not ideal machine");
        throw std::runtime_error("m_machine is not ideal machine");
    }
    auto measure_qubits = m_qlist;
    measure_qubits.pop_back();
    auto result = temp->PMeasure(measure_qubits, -1);

    std::fstream fout;
    fout.open(m_log_file + "_measure.txt", std::ios::out);

    std::cout << "Measure result: " << std::endl;
    for (auto& i : result)
    {
        if (fabs(i.second) < 1e-4)
        {
            break;
        }
        std::cout << i.first << " " << i.second << std::endl;
        fout << i.first << " " << i.second << std::endl;
    }

    fout.close();

    return result;
}

void QITE::initEnvironment()
{
    if (!m_log_file.empty())
    {
        m_log_writer.open(m_log_file, std::ios::out);
    }
    else
    {
        m_log_writer.open("test.log", std::ios::out);
    }

    m_machine.reset(QuantumMachineFactory::GetFactoryInstance()
        .CreateByType(m_quantum_machine_type));
    m_machine->init();

    m_theta_index_vec.clear();
    m_hamiltonian = m_pauli.toHamiltonian();
    auto qubits_num = 0;
    for (int i = 0; i < m_ansatz.size(); i++)
    {
        if (m_ansatz[i].target > qubits_num)
        {
            qubits_num = m_ansatz[i].target;
        }

        if (m_ansatz[i].control > qubits_num)
        {
            qubits_num = m_ansatz[i].control;
        }

        if (m_ansatz[i].type == AnsatzGateType::AGT_RX ||
            m_ansatz[i].type == AnsatzGateType::AGT_RY ||
            m_ansatz[i].type == AnsatzGateType::AGT_RZ)
        {
            m_theta_index_vec.push_back(i);
        }
    }
    qubits_num++;

    m_best_ansatz = m_ansatz;
    m_qlist = m_machine->allocateQubits(qubits_num+1);
    m_expectation = getExpectation(m_ansatz);
}

double QITE::getExpectation(const std::vector<AnsatzGate>& ansatz)
{
    QCircuit circuit = constructCircuit(ansatz);
    double expectation = 0.0;
    for (size_t i = 0; i < m_hamiltonian.size(); i++)
    {
        expectation += getExpectationOneTerm(circuit, m_hamiltonian[i]);
    }

    return expectation;
}

double QITE::getExpectationOneTerm( QCircuit c, const QHamiltonianItem& component)
{
    if (component.first.empty())
    {
        return component.second;
    }

    QProg prog;
    prog << c;
    
    for (auto iter : component.first)
    {
        if (iter.second == 'X')
        {
            prog << H(m_qlist[iter.first]);
        }
        else if (iter.second == 'Y')
        {
            prog << RX(m_qlist[iter.first], PI / 2);
        }
    }

    m_machine->directlyRun(prog);
    double expectation = 0;

    auto temp = dynamic_cast<IdealMachineInterface*>(m_machine.get());
    if (nullptr == temp)
    {
        QCERR("m_machine is not ideal machine");
        throw std::runtime_error("m_machine is not ideal machine");
    }
    auto measure_qubit = m_qlist;
    measure_qubit.pop_back();
    auto result = temp->PMeasure(measure_qubit, -1);

    for (auto i = 0u; i < result.size(); i++)
    {
        if (ParityCheck(result[i].first, component.first))
        {
            expectation -= result[i].second;
        }
        else
        {
            expectation += result[i].second;
        }
    }

    return expectation * component.second;
}

bool QITE::ParityCheck(size_t state, const QTerm& paulis) const
{
    size_t check = 0;
    for (auto iter = paulis.begin(); iter != paulis.end(); iter++)
    {
        auto value = state >> iter->first;
        if ((value % 2) == 1)
        {
            check++;
        }
    }

    return 1 == check % 2;
}

void QITE::calcParaA()
{
    int theta_num = m_theta_index_vec.size();
    m_A = Eigen::MatrixXd(theta_num, theta_num);
    for (int i = 0; i < theta_num; i++)
    {
        for (int j = 0; j < theta_num; j++)
        {
            auto t_i = m_theta_index_vec[i];
            auto t_j = m_theta_index_vec[j];
            if (i > j)
            {
                m_A.row(i)[j] = m_A.row(j)[i];
                continue;
            }

            int k = getAnsatzDerivativeParaNum(t_i);
            int l = getAnsatzDerivativeParaNum(t_j);

            double sum = 0;
            for (int p = 0; p < k; p++)
            {
                for (int q = 0; q < l; q++)
                {
                    auto f_ik = getAnsatzDerivativePara(t_i, p);
                    auto f_jl = getAnsatzDerivativePara(t_j, q);
                    auto ff = complexDagger(f_ik) * f_jl;
                    auto die_len = std::abs(ff);

                    if ((i ==j) &&(p == q))
                    {
                        // Identity
                        sum += die_len;
                    }
                    else
                    {
                        auto phase = std::arg(ff);
                        auto value = calcSubCircuit(
                            t_i,
                            t_j,
                            phase,
                            getAnsatzDerivativeCircuit(t_i, p),
                            getAnsatzDerivativeCircuit(t_j, q));

                        sum += die_len * value;
                    }
                }
            }

            m_A.row(i)[j] = sum;
        }
    }
}

void QITE::calcParaC()
{
    int theta_num = m_theta_index_vec.size();
    m_C = Eigen::VectorXd(theta_num);
    for (int i = 0; i < theta_num; i++)
    {
        auto t_i = m_theta_index_vec[i];
        double sum = 0;
        int k = getAnsatzDerivativeParaNum(t_i);
        int l = getHamiltonianItemNum();
        for (int p = 0; p < k; p++)
        {
            for (int q = 0; q < l; q++)
            {
                auto f_ik = getAnsatzDerivativePara(t_i, p);
                auto f_l = getHamiltonianItemPara(q);
                auto ff = complexDagger(f_ik) * f_l;
                auto die_len = std::abs(ff);
                auto phase = std::arg(ff);

                auto value = calcSubCircuit(
                    t_i,
                    m_ansatz.size(),
                    phase,
                    getAnsatzDerivativeCircuit(t_i, p),
                    getHamiltonianItemCircuit(q));

                sum += die_len * value;
            }
        }
        m_C[i] = sum;
    }
    m_C *= -1;
}

QCircuit QITE::constructCircuit(const std::vector<AnsatzGate>& ansatz)
{
    QCircuit circuit;
    for (int i = 0; i < ansatz.size(); i++)
    {
        circuit << convertAnsatzToCircuit(ansatz[i]);
    }

    return circuit;
}

std::complex<double> QITE::complexDagger(std::complex<double>& value)
{
    return std::complex<double>(value.real(), -value.imag());
}

int QITE::getAnsatzDerivativeParaNum(int i)
{
    if (i < 0 ||
        i >= m_ansatz.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error, 
            "bad para of i in getAnsatzDerivativeParaNum");
    }

    auto& u = m_ansatz[i];

    return u.control == -1 ? 1 : 2;
}

int QITE::getHamiltonianItemNum()
{
    return m_hamiltonian.size();
}

std::complex<double> QITE::getAnsatzDerivativePara(int i, int cnt)
{
    if (i < 0 ||
        i >= m_ansatz.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error,
            "bad para of i in getAnsatzDerivativePara");
    }

    auto& u = m_ansatz[i];

    if (u.control != -1)
    {
        if (cnt < 0 ||
            cnt > 1)
        {
            QCERR_AND_THROW_ERRSTR(std::runtime_error,
                "bad para of cnt in getAnsatzDerivativePara");
        }

        return cnt == 0 ? std::complex<double>(0, -0.25) :
            std::complex<double>(0, 0.25);
    }
    else
    {
        if (cnt != 0)
        {
            QCERR_AND_THROW_ERRSTR(std::runtime_error,
                "bad para of cnt in getAnsatzDerivativePara");
        }

        return std::complex<double>(0, -0.5);
    }
}

double QITE::getHamiltonianItemPara(int i)
{
    if (i < 0 ||
        i >= m_hamiltonian.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error,
            "bad para of i in getHamiltonianItemPara");
    }

    return m_hamiltonian[i].second;
}

QCircuit QITE::getAnsatzDerivativeCircuit(int i, int cnt)
{
    if (i < 0 ||
        i >= m_ansatz.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error,
            "bad para of i in getAnsatzDerivativePara");
    }

    QCircuit sub_cir;
    int anc = m_qlist.size() - 1;

    auto& u = m_ansatz[i];

    if (u.control != -1)
    {
        if (cnt < 0 ||
            cnt > 1)
        {
            QCERR_AND_THROW_ERRSTR(std::runtime_error,
                "bad para of cnt in getAnsatzDerivativePara");
        }

        sub_cir << (cnt == 0 ? I(m_qlist[u.control]) : Z(m_qlist[u.control]));
    }

    switch (u.type)
    {
    case AnsatzGateType::AGT_RX:
        sub_cir << X(m_qlist[u.target]);
        break;
    case AnsatzGateType::AGT_RY:
        sub_cir << Y(m_qlist[u.target]);
        break;
    case AnsatzGateType::AGT_RZ:
        sub_cir << Z(m_qlist[u.target]);
        break;
    }

    return sub_cir.control({ m_qlist[anc] });
}

QCircuit QITE::getHamiltonianItemCircuit(int cnt)
{
    if (cnt < 0 ||
        cnt >= m_hamiltonian.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error,
            "bad para of cnt in getHamiltonianItemPara");
    }

    QCircuit cir;
    auto term = m_hamiltonian[cnt].first;

    for (auto& iter : term)
    {
        switch (iter.second)
        {
        case 'X':
            cir << X(m_qlist[iter.first]);
            break;
        case 'Y':
            cir << Y(m_qlist[iter.first]);
            break;
        case 'Z':
            cir << Z(m_qlist[iter.first]);
            break;
        default:
            cir << I(m_qlist[iter.first]);
            break;
        }
    }

    return cir.control({ m_qlist[m_qlist.size() - 1] });
}

double QITE::calcSubCircuit(
    int index1, 
    int index2, 
    double theta, 
    QCircuit cir1, 
    QCircuit cir2)
{
    auto anc_quibit = m_qlist[m_qlist.size() - 1];

    QProg prog;
    for (int i = 0; i < index1; i++)
    {
        prog << convertAnsatzToCircuit(m_ansatz[i]);
    }

    prog << H(anc_quibit)
        << U1(anc_quibit, theta)
        << X(anc_quibit)
        << cir1
        << X(anc_quibit);

    for (int i = index1; i < index2; i++)
    {
        prog << convertAnsatzToCircuit(m_ansatz[i]);
    }

    prog << cir2
        << H(anc_quibit);

    auto temp = dynamic_cast<IdealMachineInterface*>(m_machine.get());
    auto result = temp->probRunDict(prog, { anc_quibit }, -1);

    return 2 * result["0"] - 1;
}

QCircuit QITE::convertAnsatzToCircuit(const AnsatzGate& u)
{
    if (u.target < 0 || u.target >= m_qlist.size())
    {
        QCERR_AND_THROW_ERRSTR(std::runtime_error,
            "bad para of target in convertAnsatzToCircuit");
    }

    QCircuit sub_cir;
    switch (u.type)
    {
    case AnsatzGateType::AGT_NOT:
        sub_cir << X(m_qlist[u.target]);
        break;
    case AnsatzGateType::AGT_H:
        sub_cir << H(m_qlist[u.target]);
        break;
    case AnsatzGateType::AGT_RX:
        sub_cir << RX(m_qlist[u.target], u.theta);
        break;
    case AnsatzGateType::AGT_RY:
        sub_cir << RY(m_qlist[u.target], u.theta);
        break;
    case AnsatzGateType::AGT_RZ:
        sub_cir << RZ(m_qlist[u.target], u.theta);
        break;
    default:
        break;
    }

    if (u.control != -1)
    {
        sub_cir.setControl({ m_qlist[u.control] });
    }

    return sub_cir;
}

Eigen::MatrixXd QITE::pseudoinverse(Eigen::MatrixXd matrix)
{
    auto svd = matrix.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& singularValues = svd.singularValues();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> 
        singularValuesInv(matrix.cols(), matrix.rows());
    singularValuesInv.setZero();
    double  pinvtoler = m_arbitary_cofficient;
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > pinvtoler)
            singularValuesInv(i, i) = 1.0 / singularValues(i);
        else
            singularValuesInv(i, i) = 0.0;
    }
    Eigen::MatrixXd pinvmat = 
        svd.matrixV() * singularValuesInv * svd.matrixU().transpose();

    return pinvmat;
}

prob_tuple qite(
    const PauliOperator& h, 
    const std::vector<AnsatzGate>& ansatz_gate, 
    size_t iter_num, 
    std::string log_file,
    QITE::UpdateMode mode, 
    size_t up_throw_num, 
    double delta_tau, 
    double convergence_factor_Q, 
    double arbitary_cofficient, 
    QMachineType type)
{
    QITE alg;
    alg.setHamiltonian(h);
    alg.setAnsatzGate(ansatz_gate);
    alg.setIterNum(iter_num);
    alg.setLogFile(log_file);
    alg.setParaUpdateMode(mode);
    alg.setUpthrowNum(up_throw_num);
    alg.setDeltaTau(delta_tau);
    alg.setConvergenceFactorQ(convergence_factor_Q);
    alg.setArbitaryCofficient(arbitary_cofficient);
    alg.setQuantumMachineType(type);

    if (alg.exec() != 0)
    {
        return prob_tuple();
    }

    return alg.getResult();
}

QPANDA_END
