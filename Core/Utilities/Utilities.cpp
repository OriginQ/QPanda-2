#include "Core/Utilities/Utilities.h"
#include <time.h>
#include "QProgram.h"
#include "ControlFlow.h"

#ifdef WIN32
#define localtime_r(_Time, _Tm) localtime_s(_Tm, _Time)
#endif

using namespace std;

USING_QPANDA
std::string QPanda::dec2bin(unsigned n, size_t size)
{
    std::string binstr = "";
    for (int i = 0; i < size; ++i)
    {
        binstr = (char)((n & 1) + '0') + binstr;
        n >>= 1;
    }
    return binstr;
}

double QPanda::RandomNumberGenerator()
{
    /*
    *  define constant number in 16807 generator.
    */
    int  ia = 16807, im = 2147483647, iq = 127773, ir = 2836;
    time_t rawtime;
    struct tm  timeinfo;
    time(&rawtime);
    localtime_r(&rawtime, &timeinfo);
    static int irandseed = timeinfo.tm_year + 70 *
        (timeinfo.tm_mon + 1 + 12 *
        (timeinfo.tm_mday + 31 *
            (timeinfo.tm_hour + 23 *
            (timeinfo.tm_min + 59 * timeinfo.tm_sec))));

    static int irandnewseed = 0;
    if (ia * (irandseed % iq) - ir * (irandseed / iq) >= 0)
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq);
    }
    else
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq) + im;
    }
    irandseed = irandnewseed;
    return (double)irandnewseed / im;
}

void QPanda::add_up_a_map(map<string, size_t> &meas_result, string key)
{
    if (meas_result.find(key) != meas_result.end())
    {
        meas_result[key]++;
    }
    else
    {
        meas_result[key] = 1;
    }
}


void QPanda::insertQCircuit(AbstractQGateNode * pGateNode, QCircuit & qCircuit, QNode * pParentNode)
{
    if ((nullptr == pParentNode) || (nullptr == pGateNode))
    {
        QCERR("param is nullptr");
        throw invalid_argument("param is nullptr");
    }

    int iNodeType = pParentNode->getNodeType();

    if (CIRCUIT_NODE == iNodeType)
    {
        auto pParentCircuit = dynamic_cast<AbstractQuantumCircuit *>(pParentNode);

        if (nullptr == pParentCircuit)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        auto aiter = pParentCircuit->getFirstNodeIter();

        if (pParentCircuit->getEndNodeIter() == aiter)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        for (; aiter != pParentCircuit->getEndNodeIter(); ++aiter)
        {
            auto temp = dynamic_cast<QNode *>(pGateNode);
            if (temp == (*aiter).get())
            {
                break;
            }
        }

        aiter = pParentCircuit->deleteQNode(aiter);

        if (nullptr == aiter.getPCur())
        {
            pParentCircuit->pushBackNode(&qCircuit);
        }
        else
        {
            pParentCircuit->insertQNode(aiter, &qCircuit);
        }

    }
    else if (PROG_NODE == iNodeType)
    {
        auto pParentQProg = dynamic_cast<AbstractQuantumProgram *>(pParentNode);

        if (nullptr == pParentQProg)
        {
            QCERR("parent node type error");
            throw invalid_argument("parent node type error");
        }

        auto aiter = pParentQProg->getFirstNodeIter();

        if (pParentQProg->getEndNodeIter() == aiter)
        {
            QCERR("unknow error");
            throw runtime_error("unknow error");
        }

        for (; aiter != pParentQProg->getEndNodeIter(); ++aiter)
        {
            auto temp = dynamic_cast<QNode *>(pGateNode);
            if (temp == (*aiter).get())
            {
                break;
            }
        }
        pParentQProg->insertQNode(aiter, &qCircuit);
        aiter = pParentQProg->deleteQNode(aiter);

    }
    else if (QIF_START_NODE == iNodeType)
    {
        auto pParentIf = dynamic_cast<AbstractControlFlowNode *>(pParentNode);

        if (nullptr == pParentIf)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        if (pGateNode ==
            dynamic_cast<AbstractQGateNode *>(pParentIf->getTrueBranch()))
        {
            pParentIf->setTrueBranch(&qCircuit);
        }
        else if (pGateNode ==
            dynamic_cast<AbstractQGateNode *>(pParentIf->getFalseBranch()))
        {
            pParentIf->setFalseBranch(&qCircuit);
        }
        else
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

    }
    else if (WHILE_START_NODE == iNodeType)
    {
        auto pParentIf = dynamic_cast<AbstractControlFlowNode *>(pParentNode);

        if (nullptr == pParentIf)
        {
            QCERR("parent if type is error");
            throw runtime_error("parent if type is error");
        }


        if (pGateNode ==
            dynamic_cast<AbstractQGateNode *>(pParentIf->getTrueBranch()))
        {
            pParentIf->setTrueBranch(&qCircuit);
        }
        else
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

}
