#include <iostream>
#include <cstdio>
#include "TestManager/TestManager.h"
#include "VQE/VQE.h"
#include "Operator/PauliOperator.h"
#include "Operator/FermionOperator.h"
#include <Eigen/Dense>

void testFermionOp()
{
    QPanda::FermionOperator a({{"1 0", 2}});
    QPanda::FermionOperator b({{"3+ 1+", 2}});
    auto x = a*b;
    auto w = x.normal_ordered();

    QPanda::FermionOperator c({{"2+ 4", 2}});
    auto y = w*c;
    auto t = y.normal_ordered();

    auto wt = w*t;

    std::cout << "x:" << x << std::endl;
    std::cout << "w:" << w << std::endl;
    std::cout << "y:" << y << std::endl;
    std::cout << "t:" << t << std::endl;
    std::cout << "wt:"<< wt << std::endl;
}

//#include "Operator/PauliOperator.h"
//#include "Variational/varpaulioperator.h"
//void testVarPauli()
//{
////    using namespace QPanda;
////    PauliOperator o1({{"Z0 X1", 2}});
////    PauliOperator o2({{"X2 Y3", 3}});

////    o1 -= o2;
////    auto o = 5 * o1;
////    std::cout << o << std::endl;

//    PauliOperator::PauliMap map{{"Z0 X1", 1}, {"X2 Y3", 2}};
//    PauliOperator c1("Z0 X1", 2);
//    PauliOperator c2("X2 Y3", 3);

////    c1 -= c2;
//    auto c = 5 * c1;
//    std::cout << c << std::endl;

//    Variational::var v1(0);
//    Variational::var v2(0);

//    complex_var cv1(v1, 0);
//    complex_var cv2(v2, 0);

//    PauliOpVar p1("Z0 X1", cv1);
//    PauliOpVar p2("X2 Y3", cv2);

////    p1 -= p2;
//    Variational::var v3(0);
//    complex_var cv3(v3, 0);
//    auto p = 5.0 + p1;

//    MatrixXd d1(1, 1);
//    d1(0, 0) = 2;
//    v1.setValue(d1);

//    MatrixXd d2(1, 1);
//    d2(0, 0) = 3;
//    v2.setValue(d2);

//    auto d = p.data();
////    MatrixXd d3(1, 1);
////    d3(0, 0) = 5;
////    v3.setValue(d3);


////    std::cout << test << std::endl;
//}

int main(int argc, char* argv[])
{
    auto instance = QPanda::TestManager::getInstance();

    if (2 != argc)
    {
        std::cout  << "Please input the config file: ";
        std::string config_fileanme;
        std::cin >> config_fileanme;

        instance->setConfigFile(config_fileanme);
    }
    else
    {
        instance->setConfigFile(argv[1]);
    }

    if (!instance->exec())
    {
        std::cout << "Do test failed." << std::endl;
    }
    else
    {
        std::cout << "Do test success." << std::endl;
    }

    return 0;
}
