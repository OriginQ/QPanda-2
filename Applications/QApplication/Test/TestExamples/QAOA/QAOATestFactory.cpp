#include "QAOATestFactory.h"
#include "QAOAParaScan.h"

namespace QPanda
{
    std::unique_ptr<AbstractQAOATest>
        QAOATestFactory::makeQAOATest(const std::string &testname)
    {
        if ("ScanPara" == testname)
        {
            return std::unique_ptr<AbstractQAOATest>(new QAOAParaScan);
        }
        else
        {
            return std::unique_ptr<AbstractQAOATest>(nullptr);
        }
    }

}
