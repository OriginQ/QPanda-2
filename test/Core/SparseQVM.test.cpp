#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <regex>
#include <ctime>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Extensions/Extensions.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/SparseQVM.h"

#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA

const std::string test_IR_2 = R"(QINIT 4
CREG 0
X q[1]
RZ q[0],(1.2142586)
RY q[0],(-1.5707963)
RZ q[0],(3.1415927)
RZ q[1],(2.785055)
RY q[1],(-1.5707963)
RZ q[1],(3.1415927)
CNOT q[0],q[1]
RZ q[1],(1.1780972)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(1.1780972)
RX q[0],(1.166348)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(3.1415927)
RY q[0],(-1.5707963)
RZ q[0],(1.9265293)
RZ q[1],(3.1415927)
RY q[1],(-1.5707963)
RZ q[1],(-2.7858597)
U1 q[0],(-5.0933388)
RZ q[0],(5.0933388)
RZ q[2],(0.79714745)
CNOT q[0],q[2]
RZ q[2],(0.073127458)
CNOT q[1],q[2]
RZ q[2],(-0.77364888)
CNOT q[0],q[2]
RZ q[2],(-1.4976689)
CNOT q[1],q[2]
RZ q[0],(-2.7488936)
RZ q[0],(-2.7488936)
RZ q[1],(-1.9634954)
RZ q[1],(-1.9634954)
CNOT q[0],q[1]
RZ q[1],(1.9634954)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.78539816)
RX q[0],(0.78539816)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(1.3744468)
RZ q[0],(1.3744468)
RZ q[1],(-2.9452431)
RZ q[1],(-2.9452431)
U1 q[0],(1.9634954)
RZ q[0],(-1.9634954)
RY q[2],(1.5707963)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RZ q[0],(-2.5525806)
RY q[0],(-1.5689674)
RZ q[0],(-3.1415927)
RZ q[1],(5.05503)
RY q[1],(-3.1297431)
CNOT q[0],q[1]
RZ q[1],(1.5662295)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.43842355)
RX q[0],(0.0047457602)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(3.1415927)
RY q[0],(-1.9386427)
RZ q[0],(1.7427486)
RZ q[1],(-3.1415927)
RY q[1],(-0.38092585)
RZ q[1],(-1.3112866)
U1 q[0],(-1.7360194)
RZ q[0],(1.7360194)
RZ q[2],(-0.66352227)
CNOT q[0],q[2]
RZ q[2],(0.74852045)
CNOT q[1],q[2]
CNOT q[0],q[2]
RZ q[2],(-1.5707963)
CNOT q[1],q[2]
RZ q[0],(-4.2591824)
RY q[0],(-1.5707963)
RZ q[0],(-1.9609946)
RZ q[1],(-3.4011024)
RY q[1],(-1.5707963)
RZ q[1],(-1.5726346)
RX q[0],(1.5707963)
CNOT q[0],q[1]
RY q[1],(-0.012651531)
RX q[0],(-1.231218)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(0.89002568)
RY q[0],(-2.5456765)
RZ q[0],(0.88742153)
RZ q[1],(0.0055183385)
RY q[1],(-1.5707963)
RZ q[1],(-6.2588063)
U1 q[0],(8.4001545)
RZ q[0],(-8.4001545)
RZ q[3],(-0.39269908)
CNOT q[0],q[3]
RZ q[3],(0.78539816)
CNOT q[1],q[3]
RZ q[3],(-1.1780972)
CNOT q[0],q[3]
RZ q[3],(0.39269908)
CNOT q[2],q[3]
RZ q[3],(-0.39269908)
CNOT q[0],q[3]
RZ q[3],(-0.78539816)
CNOT q[1],q[3]
RZ q[3],(-0.39269908)
CNOT q[0],q[3]
RZ q[3],(0.39269908)
CNOT q[2],q[3]
RZ q[0],(-0.90326193)
RY q[0],(-1.8686868)
RZ q[0],(4.6584669)
RZ q[1],(2.2329676)
RY q[1],(-0.92060692)
RZ q[1],(1.5707963)
CNOT q[0],q[1]
RZ q[1],(1.1193333)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.69504456)
RX q[0],(0)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(-4.712389)
RY q[0],(-2.2123422)
RZ q[0],(1.344145)
RZ q[1],(4.712389)
RY q[1],(-0.49611708)
RZ q[1],(0.56302899)
U1 q[0],(-2.1354707)
RZ q[0],(2.1354707)
RZ q[2],(-0.22072376)
CNOT q[0],q[2]
CNOT q[1],q[2]
RZ q[2],(-1.5707963)
CNOT q[0],q[2]
RZ q[2],(0.6143538)
CNOT q[1],q[2]
RZ q[0],(0.2266513)
RY q[0],(-1.4517386)
RZ q[0],(3.1415927)
RZ q[1],(-0.52658918)
RY q[1],(-2.4901524)
CNOT q[0],q[1]
RZ q[1],(1.6276096)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.85779146)
RX q[0],(0.40384931)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RY q[0],(-3.0945441)
RZ q[0],(-0.56241115)
RZ q[1],(-3.1415927)
RY q[1],(-2.0473239)
RZ q[1],(0.9063618)
U1 q[0],(-3.5830402)
RZ q[0],(3.5830402)
RY q[2],(1.5707963)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RZ q[0],(3.0208282)
RY q[0],(-1.5707963)
RZ q[1],(1.9122845)
RY q[1],(-1.5707963)
CNOT q[0],q[1]
H q[1]
RX q[0],(-0.12322689)
H q[1]
CNOT q[0],q[1]
RY q[0],(-1.5707963)
RZ q[0],(1.4035053)
RY q[1],(-1.5707963)
RZ q[1],(-0.053432734)
U1 q[0],(-2.1354707)
RZ q[0],(2.1354707)
RZ q[2],(-0.56467441)
CNOT q[0],q[2]
RZ q[2],(0.95737349)
CNOT q[1],q[2]
RZ q[2],(-0.56467441)
CNOT q[0],q[2]
RZ q[2],(-1.398821)
CNOT q[1],q[2]
RZ q[0],(1.5707963)
RY q[0],(-1.5707963)
RZ q[1],(-1.5707963)
RY q[1],(-1.5707963)
CNOT q[0],q[1]
RZ q[1],(0.78539816)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.78539816)
RX q[0],(5.2318027e-16)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(-3.1415927)
RY q[0],(-1.5707963)
RZ q[0],(3.1415927)
RZ q[1],(-3.1415927)
RY q[1],(-1.5707963)
RZ q[1],(-3.1415927)
U1 q[0],(-6.2831853)
RZ q[0],(6.2831853)
RY q[3],(0.78539816)
CNOT q[0],q[3]
CNOT q[1],q[3]
CNOT q[0],q[3]
RY q[3],(0.78539816)
CNOT q[2],q[3]
RY q[3],(0.78539816)
CNOT q[0],q[3]
CNOT q[1],q[3]
CNOT q[0],q[3]
RY q[3],(0.78539816)
CNOT q[2],q[3]
RZ q[0],(1.0294374)
RY q[0],(-1.3690761)
RZ q[0],(-0.30618728)
RZ q[1],(3.4239852)
RY q[1],(-1.987105)
RZ q[1],(-0.24137528)
CNOT q[0],q[1]
RZ q[1],(1.7991751)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.93273231)
RX q[0],(0.088759238)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(0.89590756)
RY q[0],(-1.6009397)
RZ q[0],(-2.5813104)
RZ q[1],(4.4653851)
RY q[1],(-2.2659192)
RZ q[1],(1.7105131)
U1 q[0],(-2.6382634)
RZ q[0],(2.6382634)
CNOT q[0],q[2]
RZ q[2],(-0.94204705)
CNOT q[1],q[2]
RZ q[2],(1.8612518)
CNOT q[0],q[2]
RZ q[2],(-0.18530497)
CNOT q[1],q[2]
RZ q[0],(3.624661)
RY q[0],(-1.8496462)
RZ q[0],(-0.88341306)
RZ q[1],(1.4481828)
RY q[1],(-2.2979829)
RZ q[1],(1.4813237)
CNOT q[0],q[1]
RZ q[1],(1.6927988)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.90059141)
RX q[0],(0.17620486)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(-2.7448948)
RY q[0],(-1.8728569)
RZ q[0],(-2.2971041)
RZ q[1],(1.9670738)
RY q[1],(-1.9382032)
RZ q[1],(3.6888305)
U1 q[0],(-6.7865145)
RZ q[0],(6.7865145)
RY q[2],(0.78539816)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RY q[2],(0.78539816)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RZ q[0],(3.4285213)
RY q[0],(-1.2530708)
RZ q[0],(1.2644411)
RZ q[1],(-0.74649036)
RY q[1],(-1.1117369)
RZ q[1],(-3.6306753)
CNOT q[0],q[1]
RZ q[1],(1.2280475)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.70568911)
RX q[0],(0.02678253)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(-4.5026585)
RY q[0],(-0.79640764)
RZ q[0],(-1.6020439)
RZ q[1],(-0.29135377)
RY q[1],(-0.68703847)
RZ q[1],(-0.52257146)
U1 q[0],(-6.9617668)
RZ q[0],(6.9617668)
RZ q[2],(0.39269908)
CNOT q[0],q[2]
RZ q[2],(1.4732218)
CNOT q[1],q[2]
RZ q[2],(1.0968086)
CNOT q[0],q[2]
RZ q[2],(0.023114867)
CNOT q[1],q[2]
RZ q[0],(2.3951463)
RY q[0],(-1.0128572)
RZ q[0],(-0.48119034)
RZ q[1],(-2.7692128)
RY q[1],(-0.49549721)
RZ q[1],(-3.1743738)
CNOT q[0],q[1]
RZ q[1],(1.7869312)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.64264812)
RX q[0],(0.31681535)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(3.2491484)
RY q[0],(-1.9781811)
RZ q[0],(0.12273358)
RZ q[1],(-0.44752153)
RY q[1],(-0.9363267)
RZ q[1],(-0.96191394)
U1 q[0],(6.5690677)
RZ q[0],(-6.5690677)
RZ q[3],(-0.39269908)
CNOT q[0],q[3]
RZ q[3],(0.91629786)
CNOT q[1],q[3]
RZ q[3],(0.39269908)
CNOT q[0],q[3]
RZ q[3],(-0.91629786)
CNOT q[2],q[3]
RZ q[3],(0.13089969)
CNOT q[0],q[3]
RZ q[3],(0.39269908)
CNOT q[1],q[3]
RZ q[3],(-0.13089969)
CNOT q[0],q[3]
RZ q[3],(-0.39269908)
CNOT q[2],q[3]
RZ q[0],(1.5859153)
RY q[0],(-1.828307)
RZ q[0],(-3.8198456)
RZ q[1],(1.4983609)
RY q[1],(-1.9457804)
RZ q[1],(-3.9412016)
CNOT q[0],q[1]
RZ q[1],(1.2329249)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.89586203)
RX q[0],(0.50577916)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(-0.40421315)
RY q[0],(-1.8943597)
RZ q[0],(0.72806877)
RZ q[1],(-0.64118132)
RY q[1],(-1.6917172)
RZ q[1],(-2.0677652)
U1 q[0],(-3.8322078)
RZ q[0],(3.8322078)
RZ q[2],(0.78539816)
CNOT q[0],q[2]
RZ q[2],(0.048056271)
CNOT q[1],q[2]
RZ q[2],(0.43867852)
CNOT q[0],q[2]
RZ q[2],(-1.7518637)
CNOT q[1],q[2]
RZ q[0],(-0.46211837)
RY q[0],(-0.6889083)
RZ q[0],(-2.4333315)
RZ q[1],(-0.25445803)
RY q[1],(-2.4493895)
RZ q[1],(0.73203626)
CNOT q[0],q[1]
RZ q[1],(1.4031154)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(1.0601473)
RX q[0],(0.44957558)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(1.1610942)
RY q[0],(-1.3683177)
RZ q[0],(-3.8243067)
RZ q[1],(3.3736157)
RY q[1],(-1.1757047)
RZ q[1],(-2.1213942)
U1 q[0],(-3.2363757)
RZ q[0],(3.2363757)
RY q[2],(0.78539816)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RY q[2],(0.78539816)
CNOT q[0],q[2]
RY q[2],(0.78539816)
CNOT q[1],q[2]
RY q[0],(-1.5707963)
RZ q[1],(1.5707963)
RY q[1],(-1.5707963)
CNOT q[0],q[1]
RZ q[1],(1.3244254)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.78539816)
RX q[0],(-3.4878685e-16)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RZ q[0],(2.0145071)
RZ q[0],(-4.2686782)
RZ q[1],(-4.7489447)
RY q[1],(-3.1415927)
RZ q[1],(0.8463393)
U1 q[0],(1.5707963)
RZ q[0],(-1.5707963)
CNOT q[0],q[2]
RZ q[2],(1.5707963)
CNOT q[1],q[2]
RZ q[2],(0.78539816)
CNOT q[0],q[2]
CNOT q[1],q[2]
RZ q[0],(2.2541711)
RY q[0],(-1.5707963)
RZ q[0],(-1.5707963)
RZ q[1],(-0.88289503)
RY q[1],(-1.5707963)
RZ q[1],(-1.5707963)
CNOT q[0],q[1]
RZ q[1],(0.78539816)
RX q[1],(1.5707963)
CNOT q[1],q[0]
RY q[1],(0.53902722)
RX q[0],(-3.4878685e-16)
CNOT q[0],q[1]
RX q[0],(-1.5707963)
RY q[0],(-1.5707963)
RZ q[0],(1.5707963)
RZ q[1],(-3.1415927)
RY q[1],(-1.5707963)
U1 q[0],(1.5707963)
RZ q[0],(-1.5707963)
)";


static bool test_vf1_1()
{

    SparseSimulator sim = SparseSimulator();

    sim.setConfig({ 128,128 });
    sim.init();
    size_t i = 100;
    auto q = sim.qAllocMany(i);
    auto c = sim.cAllocMany(i);
    auto prog = QProg();
    prog << H(q[0]);
    for (int j = 0; j < i - 1; j++)
    {
        prog << CNOT(q[j], q[j + 1]);
    }

    auto res = sim.runWithConfiguration(prog,c,1000);
    for (auto &r : res)
    {
        std::cout << r.first << ", " << r.second << std::endl;
    }

    return true;

}

static bool test_vf1_2()
{

    SparseSimulator sim = SparseSimulator();

    sim.setConfig({ 128,128 });
    sim.init();
    size_t i = 100;
    auto q = sim.qAllocMany(i);
    auto c = sim.cAllocMany(i);
    auto prog1 = convert_originir_to_qprog("D://data.ir", &sim);

    std::cout << prog1 << std::endl;
    prog1 << MeasureAll(q, c);


    auto res = sim.probRunDict(prog1);
    for (auto &r : res)
    {
        std::cout << r.first << ", " << r.second << std::endl;
    }

    return true;

}



TEST(SparseQVM, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_vf1_1();
        test_val = test_vf1_2();
    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }

    //ASSERT_TRUE(test_val);

    //cout << "VF2 test over, press Enter to continue." << endl;
    //getchar();
}

#endif