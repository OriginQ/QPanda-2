#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

static void test1()
{
    NoiseQVM qm;
    qm.init();
    auto q = qm.qAllocMany(2);
    auto c = qm.cAllocMany(2);
    qm.set_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::HADAMARD_GATE, 0.02);
    qm.set_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.05);

    QProg prog;
    prog << H(q[0])
        << CNOT(q[0], q[1])
        ;

    auto density = state_tomography_density(prog, q, &qm, 1024);
    for (size_t i = 0; i < density.size(); i++)
    {
        for (auto &value : density[i])
        {
            cout << value << "\t";
        }
        cout << endl;
    }

    return;
}


static void test2()
{
    CPUQVM qm;
    qm.init();
    auto q = qm.qAllocMany(2);
    auto c = qm.cAllocMany(2);

    std::vector<std::map<std::string, double>> probs;
    std::map<std::string, double> prog_result;
    prog_result = {
        {"00", 0.480469},
        {"01", 0.0166016},
        {"10", 0.0107422},
        {"11", 0.492188}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.248047},
        {"01", 0.259766},
        {"10", 0.251953},
        {"11", 0.240234}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.24707},
        {"01", 0.258789},
        {"10", 0.244141},
        {"11", 0.25}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.273438 },
        {"01", 0.217773},
        {"10", 0.268555},
        {"11", 0.240234}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.503906},
        {"01", 0.0166016},
        {"10", 0.0253906},
        {"11", 0.454102}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.228516},
        {"01", 0.257813},
        {"10", 0.282227},
        {"11", 0.231445}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.230469},
        {"01", 0.262695},
        {"10", 0.251953},
        {"11", 0.254883}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.261719},
        {"01", 0.253906},
        {"10", 0.239258},
        {"11", 0.245117}
    };
    probs.push_back(prog_result);

    prog_result = {
        {"00", 0.0117188},
        {"01", 0.476563},
        {"10", 0.492188},
        {"11", 0.0195313}
    };
    probs.push_back(prog_result);

    auto density = state_tomography_density(2, probs);
    for (size_t i = 0; i < density.size(); i++)
    {
        for (auto &value : density[i])
        {
            cout << value << "\t";
        }
        cout << endl;
    }
    return;
}
TEST(QuantumStateTomography, test)

{
    bool test_val = false;
    try
    {
        cout << "test1" << endl;
        test1();
        cout << "test2" << endl;
        test2();
    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }
    
}
