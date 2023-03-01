#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/Optimizer/OriginBasicOptNL.h"
#include <iostream>
//#include <QPanda.h>
#define INF HUGE_VAL

using QResultPair = std::pair<std::string, double>;
using vector_d = std::vector<double>;

QResultPair utility(vector_d x, vector_d& grad, int iter, int fcall)
{
    if (grad.size()) {
        grad[0] = 2.0 * x[0] + x[1] - 4.0;
        grad[1] = 2.0 * x[1] + x[0] - 7.0;
    }
    double f = (x[0] - 2) * (x[0] - 2) + (x[1] - 3.5) * (x[1] - 3.5) + x[0] * x[1];
    return QResultPair("test", f);
}

int main(void)
{
    QPanda::OriginBasicOptNL optimizer(QPanda::OptimizerType::COBYLA);
    std::vector<double> init_para{ 0, 0 }, lb{ 0,0 }, ub{ 2,5 };
    optimizer.set_lower_and_upper_bounds(lb, ub);
    optimizer.setXatol(1e-7);
    optimizer.setFatol(1e-12);
    optimizer.setMaxFCalls(200);
    optimizer.setMaxIter(200);
    //optimizer.add_equality_constraint(inconstraint);
    optimizer.setDisp(true);
    optimizer.registerFunc(utility, init_para);
    optimizer.exec();
    auto result = optimizer.getResult();
    bool flag = abs(result.para[0] - 1. / 3) < 1e-5;
    std::cout << "             " << std::boolalpha << flag << std::endl;
    return 0;
}