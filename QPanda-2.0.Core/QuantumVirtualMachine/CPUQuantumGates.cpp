/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "config.h"

#ifndef USE_CUDA

#include "CPUQuantumGates.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
#include <omp.h>
//#include <omp.h>

using namespace std;

REGISTER_GATE_MATRIX(X, 0, 1, 1, 0)
REGISTER_GATE_MATRIX(Hadamard, SQ2, SQ2, SQ2, -SQ2)
REGISTER_GATE_MATRIX(Y, 0, -iunit, iunit, 0)
REGISTER_GATE_MATRIX(Z, 1, 0, 0, 1)
REGISTER_GATE_MATRIX(T, 1, 0, 0, SQ2 + iunit * SQ2)
REGISTER_GATE_MATRIX(S, 1, 0, 0, iunit)
REGISTER_ANGLE_GATE_MATRIX(RX_GATE, 1, 0, 0)
REGISTER_ANGLE_GATE_MATRIX(RY_GATE, 0, 1, 0)
REGISTER_ANGLE_GATE_MATRIX(RZ_GATE, 0, 0, 1)

typedef vector<complex<double>> QStat;

CPUQuantumGates::CPUQuantumGates()
{
}

CPUQuantumGates::~CPUQuantumGates()
{
    qbit2stat.clear();
}
CPUQuantumGates::CPUQuantumGates(size_t qubitSumNumber) :qbit2stat(qubitSumNumber)
{

}
QGateParam& CPUQuantumGates::findgroup(size_t qn)
{
    for (auto iter = qbit2stat.begin(); iter != qbit2stat.end(); ++iter)
    {
        if (iter->enable == false) continue;
        if (find(iter->qVec.begin(), iter->qVec.end(), qn) != iter->qVec.end()) return *iter;
    }
    throw exception();
}


/*****************************************************************************************************************
Name:        getQState
Description: get quantum state
Argin:       pQuantumProParam       quantum program prarm pointer
Argout:      sState                 string state
return:      quantum error
*****************************************************************************************************************/
//bool X86QuantumGates::getQState(string & sState,QuantumGateParam *pQuantumProParam)
//{
//    stringstream ssTemp;
//    int i = 0;
//    for (auto aiter : mvQuantumStat)
//    {
//        ssTemp << "state[" << i << "].real = " 
//               << aiter.real() << " " 
//               << "state[" << i << "].imag = "
//               << aiter.imag() << "\n";
//        i++;
//    }
//    
//    sState.append(ssTemp.str());
//    return true;
//}

/*****************************************************************************************************************
Name:        probcompare
Description: prob compare
Argin:       a        pair
b        pair
Argout:      None
return:      true or false
*****************************************************************************************************************/
bool probcompare(pair<size_t, double> a, pair<size_t, double> b)
{
    return a.second > b.second;
}

/*****************************************************************************************************************
Name:        pMeasure
Description: pMeasure gate
Argin:       qnum        qubit bit number vector
mResult     reuslt vector
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError CPUQuantumGates::pMeasure(Qnum& qnum, vector<pair<size_t, double>> &mResult, int select_max)
{
    mResult.resize(1ull << qnum.size());
    QGateParam& group0 = findgroup(qnum[0]);
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        TensorProduct(group0, findgroup(*iter));
    }

    for (size_t i = 0; i < 1ull << qnum.size(); i++)
    {
        mResult[i].first = i;
        mResult[i].second = 0;
    }
    Qnum qvtemp;
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        qvtemp.push_back(find(group0.qVec.begin(), group0.qVec.end(), *iter) - group0.qVec.begin());
    }

    for (size_t i = 0; i < group0.qstate.size(); i++)
    {
        size_t idx = 0;
        for (size_t j = 0; j < qvtemp.size(); j++)
        {
            idx += (((i >> (qvtemp[j])) % 2) << (qvtemp.size() - 1 - j));
        }
        mResult[idx].second += abs(group0.qstate[i])*abs(group0.qstate[i]);
    }

	if (select_max == -1)
	{
		sort(mResult.begin(), mResult.end(), probcompare);
		return qErrorNone;
	}
	else if(mResult.size() <= select_max)
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        return qErrorNone;
    }
    else
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        mResult.erase(mResult.begin() + select_max, mResult.end());
    }

    return qErrorNone;
}

/**
Note: Added by Agony5757, it is a simplified version of pMeasure and with
more efficiency.
**/
QError CPUQuantumGates::pMeasure(Qnum& qnum, vector<double> &mResult)
{
	mResult.resize(1ull << qnum.size());
	QGateParam& group0 = findgroup(qnum[0]);
	for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
	{
		TensorProduct(group0, findgroup(*iter));
	}
		
	Qnum qvtemp;
	for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
	{
		qvtemp.push_back(find(group0.qVec.begin(), group0.qVec.end(), *iter) - group0.qVec.begin());
	}
	
	for (size_t i = 0; i < group0.qstate.size(); i++)
	{
		size_t idx = 0;
		for (size_t j = 0; j < qvtemp.size(); j++)
		{
			idx += (((i >> (qvtemp[j])) % 2) << (qvtemp.size() - 1 - j));
		}
		mResult[idx] += group0.qstate[i].real()*group0.qstate[i].real()
			+ group0.qstate[i].imag()*group0.qstate[i].imag();
	}

	return qErrorNone;
}



/*****************************************************************************************************************
Name:        qubitMeasure
Description: measure qubit and the state collapsed
Argin:       qn    qubit number of the measurement
Argout:      None
return:      quantum error
*****************************************************************************************************************/
bool CPUQuantumGates::qubitMeasure(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    double dprob(0);

    for (size_t i = 0; i< qgroup.qstate.size(); i += ststep * 2)
    {
        for (size_t j = i; j<i + ststep; j++)
        {
            dprob += abs(qgroup.qstate[j])*abs(qgroup.qstate[j]);
        }
    }
    int ioutcome(0);

    float fi = (float)randGenerator();

    if (fi> dprob)
    {
        ioutcome = 1;
    }

    /*
    *  POVM measurement
    */
    if (ioutcome == 0)
    {
        dprob = 1 / sqrt(dprob);

        size_t j;
        //#pragma omp parallel for private(j)
        for (size_t i = 0; i < qgroup.qstate.size(); i = i + 2 * ststep)
        {
            for (j = i; j < i + ststep; j++)
            {
                qgroup.qstate[j] *= dprob;
                qgroup.qstate[j + ststep] = 0;
            }
        }
    }
    else
    {
        dprob = 1 / sqrt(1 - dprob);

        size_t j;
        //#pragma omp parallel for private(j)
        for (size_t i = 0; i < qgroup.qstate.size(); i = i + 2 * ststep)
        {
            for (j = i; j<i + ststep; j++) {
                qgroup.qstate[j] = 0;
                qgroup.qstate[j + ststep] *= dprob;
            }
        }
    }
    return ioutcome;
}


/*****************************************************************************************************************
Name:        initState
Description: initialize the quantum state
Argin:       stNumber  Quantum number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError CPUQuantumGates::initState(QuantumGateParam * param)
{
    /* if (nullptr == qbit2stat.begin())
    {
    return undefineError;
    }*/

    /*size_t stQuantumStat = (size_t)pow(2, pQuantumProParam->mQuantumBitNumber);

    try
    {
    mvQuantumStat.resize(stQuantumStat);
    }
    catch (const std::exception &e)
    {
    std::cout << e.what() << std::endl;
    return undefineError;
    }
    mvQuantumStat[0] = 1;*/
    qbit2stat.erase(qbit2stat.begin(),qbit2stat.end());
    qbit2stat.resize(param->mQuantumBitNumber);
    for (auto i = 0; i<param->mQuantumBitNumber; i++)
    {
        qbit2stat[i].qVec.push_back(i);
        qbit2stat[i].qstate.push_back(1);
        qbit2stat[i].qstate.push_back(0);
        qbit2stat[i].qubitnumber = 1;
    }

    for (auto iter = qbit2stat.begin(); iter != qbit2stat.end(); iter++)
    {
        for (auto iter1 = (*iter).qstate.begin(); iter1 != (*iter).qstate.end(); iter1++)
        {
            *iter1 = 0;
        }
        (*iter).qstate[0] = 1;
    }
    return qErrorNone;
}



///*****************************************************************************************************************
//Name:        getCalculationUnitType
//Description: compare calculation unit type
//Argin:       sCalculationUnitType   external calculation unit type
//Argout:      None
//return:      comparison results
//*****************************************************************************************************************/
//bool X86QuantumGates :: compareCalculationUnitType(string& sCalculationUnitType)
//{
//    bool bResult = false;
//
//    if (0 == sCalculationUnitType.compare(this->sCalculationUnitType))
//    {
//        bResult = true;
//    }
//    else
//    {
//        bResult = false;
//    }
//
//    return bResult;
//}
//
//
//
/*****************************************************************************************************************
Name:        endGate
Description: end gate
Argin:       pQuantumProParam       quantum program param pointer
pQGate                 quantum gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError CPUQuantumGates::endGate(QuantumGateParam * pQuantumProParam, QuantumGates * pQGate)
{

    /* vQParam qtemp;
    for (auto iter = qbit2stat.begin(); iter != qbit2stat.end(); iter++)
    {
    for (auto iter1 = (*iter).qVec.begin(); iter1 != (*iter).qVec.end(); iter++)
    {

    }
    }*/
    return qErrorNone;
}



QError  CPUQuantumGates::
unitarySingleQubitGate(size_t qn,
    QStat& matrix,
    bool isConjugate,
    double error_rate)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    if (isConjugate)
    {
        qcomplex_t temp;
        temp = matrix[1];
        matrix[1] = matrix[2];
        matrix[2] = temp;  //convert
        for (size_t i = 0; i < 3; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//dagger
    }
#pragma omp parallel for private(j,alpha,beta)
    for (long long i = 0; i < (long long)qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            alpha = qgroup.qstate[j];
            beta = qgroup.qstate[j + ststep];
            qgroup.qstate[j] = matrix[0] * alpha + matrix[1] * beta;         /* in j,the goal qubit is in |0>        */
            qgroup.qstate[j + ststep] = matrix[2] * alpha + matrix[3] * beta;         /* in j+ststep,the goal qubit is in |1> */
        }
    }
    return qErrorNone;
}

QError  CPUQuantumGates::
controlunitarySingleQubitGate(size_t qn,
    Qnum& vControlBit,
    QStat & matrix,
    bool isConjugate,
    double error_rate)
{
    if (randGenerator() > error_rate)
    {
        QGateParam& qgroup0 = findgroup(qn);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
        size_t x;

        size_t n = qgroup0.qVec.size();
        size_t ststep = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn)
            - qgroup0.qVec.begin());
        size_t index = 0;
        size_t block = 0;

        qcomplex_t alpha, beta;
        if (isConjugate)
        {
            qcomplex_t temp;
            temp = matrix[1];
            matrix[1] = matrix[2];
            matrix[2] = temp;  //×ªÖÃ
            for (size_t i = 0; i < 3; i++)
            {
                matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//¹²éî
        }

        Qnum qvtemp;
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                - qgroup0.qVec.begin());
            block += 1ull << stemp;
            qvtemp.push_back(stemp);
        }
        sort(qvtemp.begin(), qvtemp.end());
        Qnum::iterator qiter;
        size_t j;
#pragma omp parallel for private(j,index,x,qiter,alpha,beta)
        for (long long i = 0; i < (long long)M; i++)
        {
            index = 0;
            x = i;
            qiter = qvtemp.begin();

            for (j = 0; j < n; j++)
            {
                while (qiter != qvtemp.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                //index += ((x % 2)*(1ull << j));
                index += ((x & 1) << j);
                x >>= 1;
            }

            /*
            * control qubits are 1,target qubit is 0
            */
            index = index + block - ststep;
            alpha = qgroup0.qstate[index];
            beta = qgroup0.qstate[index + ststep];
            qgroup0.qstate[index] = alpha * matrix[0] + beta * matrix[1];
            qgroup0.qstate[index + ststep] = alpha * matrix[2] + beta * matrix[3];
        }
    }
    return qErrorNone;
}


QError  CPUQuantumGates::
unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    double error_rate)
{
    if (randGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        if (qgroup0.qVec[0] != qgroup1.qVec[0])
        {
            TensorProduct(qgroup0, qgroup1);
        }
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep2 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        size_t stemp1 = (ststep1>ststep2) ? ststep1 : ststep2;
        size_t stemp2 = (ststep1>ststep2) ? ststep2 : ststep1;

        bool bmark = true;
        qcomplex_t phi00, phi01, phi10, phi11;
        auto stateSize = qgroup0.qstate.size();

        if (isConjugate)
        {
            qcomplex_t temp;
            for (size_t i = 0; i < 4; i++)
            {
                for (size_t j = i + 1; j < 4; j++)
                {
                    temp = matrix[4 * i + j];
                    matrix[4 * i + j] = matrix[4 * j + i];
                    matrix[4 * j + i] = temp;
                }
            }
            for (size_t i = 0; i < 16; i++)
            {
                //matrix[i].imag = -matrix[i].imag;
                matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }
        long long j, k;
#pragma omp parallel for private(j,k,phi00,phi01,phi10,phi11)
        for (long long i = 0; i<(long long)stateSize; i = i + 2 * stemp1)
        {
            for (j = i; j <(long long)(i + stemp1); j = j + 2 * stemp2)
            {
                for (k = j; k < (long long)(j + stemp2); k++)
                {
                    phi00 = qgroup0.qstate[k];        //00
                    phi01 = qgroup0.qstate[k + ststep2];  //01
                    phi10 = qgroup0.qstate[k + ststep1];  //10
                    phi11 = qgroup0.qstate[k + ststep1 + ststep2]; //11

                    qgroup0.qstate[k] = matrix[0] * phi00 + matrix[1] * phi01
                        + matrix[2] * phi10 + matrix[3] * phi11;
                    qgroup0.qstate[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                        + matrix[6] * phi10 + matrix[7] * phi11;
                    qgroup0.qstate[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                        + matrix[10] * phi10 + matrix[11] * phi11;
                    qgroup0.qstate[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                        + matrix[14] * phi10 + matrix[15] * phi11;
                }
            }
        }

    }
    return qErrorNone;
}


QError  CPUQuantumGates::
controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    QStat& matrix,
    bool isConjugate,
    double error_rate)
{
    if (randGenerator() > error_rate)
    {
        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        qcomplex_t temp;
        if (isConjugate)
        {
            for (size_t i = 0; i < 4; i++)
            {
                for (size_t j = i + 1; j < 4; j++)
                {
                    temp = matrix[4 * i + j];
                    matrix[4 * i + j] = matrix[4 * j + i];
                    matrix[4 * j + i] = temp;
                }
            }
            for (size_t i = 0; i < 16; i++)
            {
                matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }
        }//dagger

         //combine all qubits;
        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());

        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        size_t block = 0;
        qcomplex_t phi00, phi01, phi10, phi11;
        Qnum qvtemp;
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                - qgroup0.qVec.begin());
            block += 1ull << stemp;
            qvtemp.push_back(stemp);
        }
        //block: all related qubits are 1,others are 0
        sort(qvtemp.begin(), qvtemp.end());
        Qnum::iterator qiter;
        size_t j;
        size_t index = 0;
        size_t x;
        size_t n = qgroup0.qVec.size();
        
#pragma omp parallel for private(j,index,x,qiter,phi00,phi01,phi10,phi11)
        for (long long i = 0; i < (long long)M; i++)
        {
            index = 0;
            x = i;
            qiter = qvtemp.begin();

            for (j = 0; j < n; j++)
            {
                while (qiter != qvtemp.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                //index += ((x % 2)*(1ull << j));
                index += ((x & 1) << j);
                x >>= 1;
            }
            index = index + block - ststep0 - ststep1;                             /*control qubits are 1,target qubit are 0 */
            phi00 = qgroup0.qstate[index];             //00
            phi01 = qgroup0.qstate[index + ststep1];   //01
            phi10 = qgroup0.qstate[index + ststep0];   //10
            phi11 = qgroup0.qstate[index + ststep0 + ststep1];  //11
            qgroup0.qstate[index] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            qgroup0.qstate[index + ststep1] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            qgroup0.qstate[index + ststep0] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            qgroup0.qstate[index + ststep0 + ststep1] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    return qErrorNone;

}

QError  CPUQuantumGates::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    if (randGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);

        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());

        /*
        * iSWAP(qn_1,qn_2) is agree with
        * iSWAP(qn_2,qn_1)
        */
        if (qn_0 < qn_1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(0);
            compll.imag(1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(-1.0);
        }

        qcomplex_t alpha, beta;

        /*
        *  traverse all the states
        */
        size_t j, k;
        //#pragma omp parallel for private(j,k,alpha,beta)
        for (size_t i = 0; i < qgroup0.qstate.size(); i = i + 2 * ststep0)
        {
            for (j = i + ststep1; j < i + ststep0; j = j + 2 * ststep1)
            {
                for (k = j; k < j + ststep1; k++)
                {
                    alpha = qgroup0.qstate[k];             //01
                    beta = qgroup0.qstate[k + sttemp];     //10

                    qgroup0.qstate[k] = cos(theta)*alpha + compll * beta*sin(theta);           /* k:|01>                               */
                    qgroup0.qstate[k + sttemp] = compll * sin(theta)* alpha + cos(theta)*beta;          /* k+sttemp:|10>                        */
                }
            }
        }
    }
    return qErrorNone;
}
QError  CPUQuantumGates::iSWAP(size_t qn_0, size_t qn_1, Qnum & vControlBit, double theta, bool isConjugate, double error_rate)
{
    if (randGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        if (qn_0 < qn_1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(0);
            compll.imag(1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(-1.0);
        }

        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
        size_t block = 0;
        Qnum qvtemp;
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                - qgroup0.qVec.begin());
            block += 1ull << stemp;
            qvtemp.push_back(stemp);
        }
        sort(qvtemp.begin(), qvtemp.end());
        Qnum::iterator qiter;
        size_t j;
        size_t index = 0;
        size_t x;
        size_t n = qgroup0.qVec.size();
        //#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        qcomplex_t alpha, beta;
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qvtemp.begin();

            for (j = 0; j < n; j++)
            {
                while (qiter != qvtemp.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += ((x & 1) << j);
                x >>= 1;
            }
            index = index + block - ststep0 - ststep1;                  /*control qubits are 1,target qubit are 0 */
            alpha = qgroup0.qstate[index + ststep1];             //01
            beta = qgroup0.qstate[index + ststep0];     //10
            qgroup0.qstate[index + ststep1] = cos(theta)*alpha + compll * beta*sin(theta);
            qgroup0.qstate[index + ststep0] = compll * sin(theta)* alpha + cos(theta)*beta;
        }
    }
    return qErrorNone;
}

QError  CPUQuantumGates::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    if (randGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);

        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());

        /*
        * iSWAP(qn_1,qn_2) is agree with
        * iSWAP(qn_2,qn_1)
        */
        if (qn_0 < qn_1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(cos(theta));
            compll.imag(sin(theta));
        }
        else
        {
            compll.real(cos(theta));
            compll.imag(-sin(theta));
        }
        /*
        *  traverse all the states
        */
        size_t j, k;
        //#pragma omp parallel for private(j,k,alpha,beta)
        for (size_t i = ststep0; i < qgroup0.qstate.size(); i = i + 2 * ststep0)
        {
            for (j = i + ststep1; j < i + ststep0; j = j + 2 * ststep1)
            {
                for (k = j; k < j + ststep1; k++)
                {
                    qgroup0.qstate[k] = compll * qgroup0.qstate[k];
                }
            }
        }
    }
    return qErrorNone;
}
QError  CPUQuantumGates::CR(size_t qn_0, size_t qn_1, Qnum & vControlBit, double theta, bool isConjugate, double error_rate)
{
    if (randGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        if (qn_0 < qn_1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(cos(theta));
            compll.imag(sin(theta));
        }
        else
        {
            compll.real(cos(theta));
            compll.imag(-sin(theta));
        }

        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
        size_t block = 0;
        Qnum qvtemp;
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                - qgroup0.qVec.begin());
            block += 1ull << stemp;
            qvtemp.push_back(stemp);
        }
        sort(qvtemp.begin(), qvtemp.end());
        Qnum::iterator qiter;
        size_t j;
        size_t index = 0;
        size_t x;
        size_t n = qgroup0.qVec.size();
        //#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qvtemp.begin();

            for (j = 0; j < n; j++)
            {
                while (qiter != qvtemp.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += ((x & 1) << j);
                x >>= 1;
            }
            qgroup0.qstate[index + block] *= compll;
        }
    }
    return qErrorNone;
}

QError CPUQuantumGates::Reset(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << (find(qgroup.qVec.begin(), qgroup.qVec.end(), qn)
        - qgroup.qVec.begin());
    //#pragma omp parallel for private(j,alpha,beta)
    for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            qgroup.qstate[j + ststep] = 0;                              /* in j+ststep,the goal qubit is in |1> */
        }
    }
    return qErrorNone;
}

#endif