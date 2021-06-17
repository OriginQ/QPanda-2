#include "gtest/gtest.h"
#include "QPanda.h"
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include <iostream>
#define INF HUGE_VAL

using QResultPair = std::pair<std::string, double>;
using vector_d = std::vector<double>;
using namespace std;
QResultPair utility(vector_d x, vector_d& grad, int iter, int fcall)
{
    if (grad.size()) {
        grad[0] = 2.0 * x[0] + x[1] - 4.0;
        grad[1] = 2.0 * x[1] + x[0] - 7.0;
    }
    double f = (x[0] - 2) * (x[0] - 2) + (x[1] - 3.5) * (x[1] - 3.5) + x[0] * x[1];
    return QResultPair("test", f);
}

bool test_Optimizer()
{
    auto optimizer = QPanda::OptimizerFactory::makeOptimizer(QPanda::OptimizerType::COBYLA);
    QPanda::vector_d init_para{ 0, 0 }, lb{ -INF,-INF }, ub{ INF,3 };
    optimizer->registerFunc(utility, init_para);
    //optimizer->set_lower_and_upper_bounds(lb, ub);
    optimizer->setXatol(1e-6);
    //optimizer->setFatol(1e-6);
    optimizer->setMaxFCalls(200);
    optimizer->setMaxIter(200);
    //optimizer->add_equality_constraint(inconstraint);
    optimizer->exec();
    auto result = optimizer->getResult();
    std::cout << result.message << std::endl;
    std::cout << "         Current function value: "
        << result.fun_val << std::endl;
    std::cout << "         Iterations: "
        << result.iters << std::endl;
    std::cout << "         Function evaluations: "
        << result.fcalls << std::endl;
    std::cout << "         Optimized para: " << std::endl;
    bool flag = abs(result.para[0] - 1. / 3) < 1e-2;
    std::cout << "             " << std::boolalpha << flag << std::endl;
    return flag;
}

TEST(Optimizer, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_Optimizer();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	ASSERT_TRUE(test_val);
}