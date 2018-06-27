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

#include "X86QuantumGates.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
//#include <omp.h>

using namespace std;

#define SQ2 0.707106781186548
#define LONG long long
typedef vector<complex<double>> QStat;
X86QuantumGates::X86QuantumGates()
{
}

X86QuantumGates::~X86QuantumGates()
{
    mvQuantumStat.clear();
}

/*****************************************************************************************************************
Name:        getQState
Description: get quantum state
Argin:       pQuantumProParam       quantum program prarm pointer
Argout:      sState                 string state
return:      quantum error
*****************************************************************************************************************/
bool X86QuantumGates::getQState(string & sState,QuantumGateParam *pQuantumProParam)
{
    stringstream ssTemp;
    int i = 0;
    for (auto aiter : mvQuantumStat)
    {
        ssTemp << "state[" << i << "].real = " 
               << aiter.real() << " " 
               << "state[" << i << "].imag = "
               << aiter.imag() << "\n";
        i++;
    }
    
    sState.append(ssTemp.str());
    return true;
}

/*****************************************************************************************************************
Name:        Hadamard
Description: Hadamard gate,the matrix is:[1/sqrt(2),1/sqrt(2);1/sqrt(2),-1/sqrt(2)]
Argin:       qn           qubit number that the Hadamard gate operates on.
             error_rate   the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::Hadamard(size_t qn, double error_rate)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep = (size_t)pow(2, qn);
        COMPLEX alpha, beta;

        /*
         *  traverse all the states
         */
        size_t j;
//#pragma omp parallel for private(j,alpha,beta)
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        {
            for (j = i; j<i + ststep; j++)
            {
                alpha                     = mvQuantumStat[j];
                beta                      = mvQuantumStat[j + ststep];
                mvQuantumStat[j]          = (alpha + beta)*SQ2;         /* in j,the goal qubit is in |0>        */
                mvQuantumStat[j + ststep] = (alpha - beta)*SQ2;         /* in j+ststep,the goal qubit is in |1> */
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        Reset
Description: reset bit gate
Argin:       qn          qubit number that the Hadamard gate operates on.
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::Reset(size_t qn)
{
	size_t  ststep = 1ull << qn;
    COMPLEX alpha, beta;

    /*
     *  traverse all the states
     */
    size_t j;
//#pragma omp parallel for private(j,alpha,beta)
    for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            mvQuantumStat[j + ststep] = 0;                              /* in j+ststep,the goal qubit is in |1> */
        }
    }
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        probcompare
Description: prob compare 
Argin:       a        pair
             b        pair
Argout:      None
return:      true or false
*****************************************************************************************************************/
bool probcompare(pair<size_t,double>& a, pair<size_t, double>& b)
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
/*
QError X86QuantumGates::pMeasure(Qnum& qnum, vector<pair<size_t, double>> &mResult)
{
	mResult.resize(1ull << qnum.size());
    for (LONG i = 0; i < (LONG)pow(2, qnum.size()); i++)
    {
        mResult[i].first  = i;
        mResult[i].second = 0;
    }

    for (size_t i = 0; i < mvQuantumStat.size(); i++)
    {
        size_t idx=0;
        for (LONG j = 0; j < (LONG)qnum.size(); j++)
        {
            idx+=(((i>>(qnum[j]))%2)<<(qnum.size()-1-j));            
        }
        mResult[idx].second+=abs(mvQuantumStat[i])*abs(mvQuantumStat[i]);
    }

    if (mResult.size() <= 10)
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        return qErrorNone;
    }
    else
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        mResult.erase(mResult.begin()+10,mResult.end());
    }

    return qErrorNone;
}
*/

/*****************************************************************************************************************
Name:        Hadamard
Description: controled-Hadamard gate
Argin:       qn                 qubit number that the Hadamard gate operates on.
             error_rate         the errorrate of the gate
             vControlBit        control bit vector
             stQuantumBitNumber quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::Hadamard(size_t  qn,
                                 double  error_rate, 
                                 Qnum    vControlBit, 
                                 size_t  stQuantumBitNumber)
{
    if (randGenerator() > error_rate)
    {
        int M = (int)(mvQuantumStat.size() / pow(2, vControlBit.size()));
        int x;

        size_t n      = stQuantumBitNumber;
        size_t ststep = (size_t)pow(2, qn);
        size_t index  = 0, block = 0;
        
        COMPLEX alpha, beta;

        sort(vControlBit.begin(), vControlBit.end());
        for (Qnum::iterator j = vControlBit.begin(); j != vControlBit.end(); j++)
        {
            block += (size_t)pow(2, *j);
        }

        Qnum::iterator qiter;

        size_t j;
//#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        for (int i = 0; i < M; i++)
        {
            index = 0;
            x     = i;
            qiter = vControlBit.begin();

            for (j = 0; j < n; j++)
            {
                while (qiter != vControlBit.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (size_t)((x % 2)*pow(2, j));
                x >>= 1;
            }

            /*
             * control qubits are 1,target qubit is 0
             */
            index                         = index + block - ststep;                             
            alpha                         = mvQuantumStat[index];
            beta                          = mvQuantumStat[index + ststep];
            mvQuantumStat[index]          = (alpha + beta)*SQ2;
            mvQuantumStat[index + ststep] = (alpha - beta)*SQ2;
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RX_GATE
Description: RX_GATE gate,quantum state rotates θ by x axis.The matric is:
             [cos(θ/2),-i*sin(θ/2);i*sin(θ/2),cos(θ/2)]
Argin:       qn          qubit number that the Hadamard gate operates on.
             theta       rotation angle
             error_rate  the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RX_GATE(size_t qn, double theta, double error_rate)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep    = (size_t)pow(2, qn );

        double dcostheta = cos(theta / 2);
        double dsintheta = sin(theta / 2);
        
        COMPLEX alpha, beta;
        COMPLEX compll(0, 1);

        /*
         *  traverse all the states
         */
        size_t j;
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        {
            for (j = i; j != i + ststep; j++)
            {
                alpha                     = mvQuantumStat[j];
                beta                      = mvQuantumStat[j + ststep];
                mvQuantumStat[j]          = alpha * dcostheta - compll * dsintheta*beta;
                mvQuantumStat[j + ststep] = beta * dcostheta - compll * dsintheta*alpha;
            }
        }
    }
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RX_GATE
Description: RX_GATE dagger gate,quantum state rotates θ by x axis.The matric is:
             [cos(θ/2),-i*sin(θ/2);i*sin(θ/2),cos(θ/2)]
Argin:       qn          qubit number that the Hadamard gate operates on.
             theta       rotation angle
             error_rate  the errorrate of the gate
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RX_GATE(size_t qn, double theta, double error_rate, int iDagger)
{
    if (!iDagger)
    {
        return qParameterError;
    }
   
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep = (size_t)pow(2, qn);

        double dcostheta = cos(theta / 2);
        double dsintheta = sin(theta / 2);
        
        COMPLEX alpha, beta;
        COMPLEX compll(0, 1);
        
        /*
         *  traverse all the states
         */
        size_t j;
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        for (j = i; j != i + ststep; j++)
        {
            alpha                     = mvQuantumStat[j];
            beta                      = mvQuantumStat[j + ststep];
            mvQuantumStat[j]          = alpha * dcostheta + compll *dsintheta*beta;
            mvQuantumStat[j + ststep] = beta * dcostheta + compll *dsintheta*alpha;
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RX_GATE
Description: controled-RX_GATE gate
Argin:       qn                 qubit number that the Hadamard gate operates on.
             theta              rotation angle
             error_rate         the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RX_GATE(size_t   qn,
                           double   theta,
                           double   error_rate,
                           Qnum     vControlBitNumber,
                           size_t   stQuantumBitNumber)
{
    if (randGenerator() > error_rate)
    {

        int x;
        int M          = (int)(mvQuantumStat.size() / pow(2, vControlBitNumber.size()));

        size_t  n      = stQuantumBitNumber;
        size_t  ststep = (size_t)pow(2, qn);
        size_t  index  = 0, block = 0;
           
        double  dcostheta = cos(theta / 2);
        double  dsintheta = sin(theta / 2);
        
        COMPLEX alpha, beta;
        COMPLEX compll(0, 1);

        sort(vControlBitNumber.begin(), vControlBitNumber.end());
        for (Qnum::iterator j = vControlBitNumber.begin(); j != vControlBitNumber.end(); j++)
        {
            block += (size_t)pow(2, *j);
        }

        Qnum::iterator qiter;

        size_t j;
//#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        for (int i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = vControlBitNumber.begin();
            for (j = 0; j < n; j++)
            {
                while (qiter != vControlBitNumber.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (size_t)((x % 2)*pow(2, j));
                x >>= 1;
            }
            index                         = index + block - ststep;
            alpha                         = mvQuantumStat[index];
            beta                          = mvQuantumStat[index + ststep];
            mvQuantumStat[index]          = alpha * dcostheta - compll *dsintheta*beta;
            mvQuantumStat[index + ststep] = beta * dcostheta - compll *dsintheta*alpha;
        }
    }
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RX_GATE
Description: controled-RX_GATE dagger gate
Argin:       qn                 qubit number that the Hadamard gate operates on.
             theta              rotation angle
             error_rate         the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
             iDagger            is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RX_GATE(size_t  qn,
                           double  theta,
                           double  error_rate,
                           Qnum    vControlBitNumber,
                           size_t  stQuantumBitNumber,
                           int     iDagger)
{
    if (!iDagger)
    {
        return qParameterError;              
    }
    if (randGenerator() > error_rate)
    {
        int M = (int)(mvQuantumStat.size() / pow(2, vControlBitNumber.size()));
        int x;

        size_t n      = stQuantumBitNumber;
        size_t ststep = (size_t)pow(2, qn);
        size_t index  = 0;
        size_t block  = 0;

        double dcostheta = cos(theta / 2);
        double dsintheta = sin(theta / 2);

        COMPLEX compll(0, 1);
        COMPLEX alpha, beta;

        sort(vControlBitNumber.begin(), vControlBitNumber.end());
        for (Qnum::iterator j = vControlBitNumber.begin(); j != vControlBitNumber.end(); j++)
        {
            block += (size_t)pow(2, *j);
        }

        Qnum::iterator qiter;
        size_t j;
//#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        for (int i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = vControlBitNumber.begin();
            for ( j = 0; j < n; j++)
            {
                while (qiter != vControlBitNumber.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += size_t((x % 2)*pow(2, j));
                x >>= 1;
            }
            index                         = index + block - ststep;
            alpha                         = mvQuantumStat[index];
            beta                          = mvQuantumStat[index + ststep];
            mvQuantumStat[index]          = alpha * dcostheta + compll *dsintheta*beta;
            mvQuantumStat[index + ststep] = beta * dcostheta + compll *dsintheta*alpha;
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RY_GATE
Description: RY_GATE control gate
Argin:       qn                 qubit number that the Hadamard gate operates on.
             theta              rotation angle
             error_rate         the errorrate of the gate
             vControlBit        control bit vector
             stQuantumBitNumber quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RY(size_t   qn,
                          double   theta,
                          double   error_rate,
                          Qnum     vControlBit,
                          size_t   stQuantumBitNumber, int iDagger)
{
    if (randGenerator() > error_rate)
    {
        int M = int(mvQuantumStat.size() / pow(2, vControlBit.size()));
        int x;

        size_t qn = vControlBit.back();
        size_t n = stQuantumBitNumber;
        size_t ststep = (size_t)pow(2, qn);
        size_t index = 0, block = 0;

        double dcostheta = cos(theta / 2);
        double dsintheta = sin(theta / 2);

        COMPLEX compll(0, 1);
        COMPLEX alpha, beta;

        sort(vControlBit.begin(), vControlBit.end());
        for (Qnum::iterator j = vControlBit.begin(); j != vControlBit.end(); j++)
        {
            block += (size_t)pow(2, *j);
        }

        Qnum::iterator qiter;
        size_t j;

        for (int i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = vControlBit.begin();
            for ( j = 0; j < n; j++)
            {
                while (qiter != vControlBit.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (size_t)((x % 2)*pow(2, j));
                x >>= 1;
            }
            index = index + block - ststep;
            alpha = mvQuantumStat[index];
            beta = mvQuantumStat[index + ststep];
            if (iDagger)
            {
                mvQuantumStat[index]          = alpha * dcostheta + compll * dsintheta*beta;
                mvQuantumStat[index + ststep] = beta * dcostheta - compll * dsintheta*alpha;
            }
            else
            {
                mvQuantumStat[index]          = alpha * dcostheta - compll * dsintheta*beta;
                mvQuantumStat[index + ststep] = beta * dcostheta + compll * dsintheta*alpha;
            }

        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RY_GATE
 
Description: RY_GATE gate,quantum state rotates θ by y axis.The matric is
             [cos(θ/2),-sin(θ/2);sin(θ/2),cos(θ/2)]
Argin:       qn          qubit number that the Hadamard gate operates on.
                         theta        rotation angle
             error_rate   the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RY(size_t qn, double theta, double error_rate, int iDagger)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        double dcostheta = cos(theta / 2);
        double dsintheta = sin(theta / 2);
        
        size_t ststep = (size_t)pow(2, qn);

        COMPLEX alpha, beta;
        /*
         *  traverse all the states
         */
        size_t j;

        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        {
            for (j = i; j != i + ststep; j++)
            {
                alpha = mvQuantumStat[j];
                beta = mvQuantumStat[j + ststep];
                if (iDagger)
                {
                    mvQuantumStat[j]          = alpha * dcostheta + dsintheta * beta;
                    mvQuantumStat[j + ststep] = beta * dcostheta - dsintheta * alpha;
                }
                else
                {
                    mvQuantumStat[j]          = alpha * dcostheta - dsintheta * beta;
                    mvQuantumStat[j + ststep] = beta * dcostheta + dsintheta * alpha;
                }
            }
        }
    }
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RZ_GATE
Description: RZ_GATE gate,quantum state rotates θ by z axis.The matric is
             [1 0;0 exp(i*θ)]
Argin:       qn          qubit number that the Hadamard gate operates on.
             theta       rotation angle
             error_rate  the errorrate of the gate
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::RZ(size_t qn, double theta, double error_rate, int iDagger)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep = (size_t)pow(2, qn);

        double dcostheta = cos(theta);
        double dsintheta = sin(theta);
        
        COMPLEX alpha, beta;
        COMPLEX compll;
        
        if (!iDagger)
        {
            compll.real(dcostheta);
            compll.imag(dsintheta);
        }
        else
        {
            compll.real(dcostheta);
            compll.imag(-dsintheta);
        }

        /*
         *  traverse all the states
         */
        size_t j;
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        {
            for (j = i; j != i + ststep; j++)
            {
                alpha                     = mvQuantumStat[j];
                beta                      = mvQuantumStat[j + ststep];
                mvQuantumStat[j]          = alpha;
                mvQuantumStat[j + ststep] = compll *mvQuantumStat[j + ststep];
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        RZ_GATE
Description: RZ_GATE gate
Argin:       qn                 qubit number that the Hadamard gate operates on.
             theta              rotation angle
             error_rate         the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
             iDagger            is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates :: RZ(size_t   qn,
                             double   theta,
                             double   error_rate,
                             Qnum     vControlBitNumber,
                             size_t   stQuantumBitNumber,
                             int      iDagger)
{
    if (randGenerator() > error_rate)
    {
        int M = (int)(mvQuantumStat.size() / pow(2, vControlBitNumber.size()));
        int x;

        size_t n = stQuantumBitNumber;
        size_t index = 0, block = 0;

        double dcostheta = cos(theta);
        double dsintheta = sin(theta);
        
        COMPLEX compll;

        sort(vControlBitNumber.begin(), vControlBitNumber.end());
        if (!iDagger)
        {
            compll.real(dcostheta);
            compll.imag(dsintheta);
        }
        else
        {
            compll.real(dcostheta);
            compll.imag(-dsintheta);
        }

        for (Qnum::iterator j = vControlBitNumber.begin(); j != vControlBitNumber.end(); j++)
        {
            block += (size_t)pow(2, *j);
        }

        Qnum::iterator qiter;
        size_t j;

        for (int i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = vControlBitNumber.begin();
            for (j = 0; j < n; j++)
            {
                while (qiter != vControlBitNumber.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (size_t)((x % 2)*pow(2, j));
                x >>= 1;
            }
            index = index + block;
            mvQuantumStat[index] = compll *mvQuantumStat[index];
        }
    }

    return qErrorNone;

}

/*****************************************************************************************************************
Name: CNOT_GATE
Description: CNOT_GATE gate,when control qubit is |0>,goal qubit does flip,
             when control qubit is |1>,goal qubit flips.the matric is:
             [1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0]
Argin:       qn_1        control qubit number
             qn_2        goal qubit number
             error_rate  the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::CNOT(size_t qn_1, size_t qn_2, double error_rate)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep1 = (size_t)pow(2, qn_1);
        size_t ststep2 = (size_t)pow(2, qn_2);
        
        bool bmark     = true;
        
        COMPLEX alpha, beta;

        if (qn_1>qn_2)                                                  /* control qubit number is higher       */
        {
            /*
             *  traverse all the states
             */
            size_t j,k;
            for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + ststep1)
            {
                bmark = !bmark;
                for (j = i; j < i + ststep1; j = j + 2 * ststep2)
                {
                    for (k = j; k < j + ststep2; k++)
                    {
                        if (bmark)
                        {
                            alpha                      = mvQuantumStat[k];
                            beta                       = mvQuantumStat[k + ststep2];
                            mvQuantumStat[k]           = beta;          /* k:control |1>,goal |0>               */
                            mvQuantumStat[k + ststep2] = alpha;         /* k+ststep:control |1>,goal |1>        */
                        }

                    }
                }
            }
        }
        else                                                            /* control qubit numer is lower         */
        {
            /*
             *  traverse all the states
             */
            size_t j, k;
            for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep2)
            {
                for (j = i; j < i + ststep2; j = j + ststep1)
                {
                    bmark = !bmark;
                    for (k = j; k < j + ststep1; k++)
                    {
                        if (bmark)
                        {
                            alpha                      = mvQuantumStat[k];
                            beta                       = mvQuantumStat[k + ststep2];
                            mvQuantumStat[k]           = beta;
                            mvQuantumStat[k + ststep2] = alpha;
                        }

                    }
                }
            }
        }
    }

    return qErrorNone;
}  

/*****************************************************************************************************************
Name:        CNOT_GATE
Description: CNOT_GATE control gate
Argin:       qn_1               control qubit number
             qn_2               goal qubit number
             error_rate         the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates :: CNOT(size_t  qn_1,
                               size_t  qn_2,
                               double  error_rate,
                               Qnum    vControlBitNumber,
                               int     stQuantumBitNumber)
{
    return qErrorNone;
}


/*****************************************************************************************************************
Name:        CR
Description: CR gate,when control qubit is |0>,goal qubit does not rotate,
             when control qubit is |1>,goal qubit rotate θ by z axis.the matric is:
             [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 exp(i*θ)]
Argin:       qn_1        control qubit number
             qn_2        goal qubit number
             theta       roration angle
             error_rate  the errorrate of the gate
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::CR(size_t  qn_1,
                           size_t  qn_2,
                           double  theta,
                           double  error_rate,
                           int     iDagger)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep1 = (size_t)pow(2, qn_1);
        size_t ststep2 = (size_t)pow(2, qn_2);
        
        bool bmark = 1;
        
        double dcostheta = cos(theta);
        double dsintheta = sin(theta);

        COMPLEX alpha, beta;
        COMPLEX compll;
        
        if (!iDagger)
        {
            compll.real(dcostheta);
            compll.imag(dsintheta);
        }
        else
        {
            compll.real(dcostheta);
            compll.imag(-dsintheta);
        }
        if (qn_1>qn_2)                                                  /* control qubit numer is higher        */
        {
            
            /*
             *  traverse all the states
             */
            size_t j, k;
            for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + ststep1)
            {
                bmark = !bmark;
                for (j = i; j < i + ststep1; j = j + 2 * ststep2)
                {
                    for (k = j; k < j + ststep2; k++)
                    {
                        if (bmark)
                        {
                            alpha                      = mvQuantumStat[k];
                            beta                       = mvQuantumStat[k + ststep2];
                            mvQuantumStat[k]           = alpha;         /* k:control |1>,goal |0>               */
                            mvQuantumStat[k + ststep2] = beta* compll;  /* k+ststep:control |1>,goal |1>        */
                        }
                    }
                }
            }
        }
        else
        {                                                               /* control qubit numer is lower         */
            /*
             *  traverse all the states
             */
            size_t j, k;
            for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep2)
            {
                for (j = i; j < i + ststep2; j = j + ststep1)
                {
                    bmark = !bmark;
                    for (k = j; k < j + ststep1; k++)
                    {
                        if (bmark)
                        {
                            alpha                      = mvQuantumStat[k];
                            beta                       = mvQuantumStat[k + ststep2];
                            mvQuantumStat[k]           = alpha;
                            mvQuantumStat[k + ststep2] = beta* compll;
                        }
                    }
                }
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        iSWAP
Description: iSWAP gate,both qubits swap and rotate π by z-axis,the matric is:
[1 0 0 0;0 0 -i 0;0 -i 0 0;0 0 0 1]
Argin:       qn_1        first qubit number
             qn_2        second qubit number
             error_rate  the errorrate of the gate
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::iSWAP(size_t qn_1, 
                              size_t qn_2,
                              double error_rate,
                              int    iDagger)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t sttemp = 0;
        size_t ststep1 = (size_t)pow(2, qn_1);
        size_t ststep2 = (size_t)pow(2, qn_2);

        /* 
         * iSWAP(qn_1,qn_2) is agree with
         * iSWAP(qn_2,qn_1)                     
         */
        if (qn_1 < qn_2)                                                
        {
            sttemp  = ststep1;
            ststep1 = ststep2;
            ststep2 = sttemp;
        }
        sttemp = ststep1 - ststep2;
        COMPLEX compll;
        if (!iDagger)
        {
            compll.real(0);
            compll.imag(-1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(1.0);
        }

        COMPLEX alpha, beta;

        /*
         *  traverse all the states
         */
        size_t j, k;
//#pragma omp parallel for private(j,k,alpha,beta)
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep1)
        {
            for (j = i + ststep2; j < i + ststep1; j = j + 2 * ststep2)
            {
                for (k = j; k < j + ststep2; k++)
                {
                    alpha  = mvQuantumStat[k];
                    beta   = mvQuantumStat[k + sttemp];
                    
                    mvQuantumStat[k]          = compll *beta;           /* k:|01>                               */
                    mvQuantumStat[k + sttemp] = compll *alpha;          /* k+sttemp:|10>                        */
                }
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        iSWAP
Description: iSWAP gate,both qubits swap and rotate π by z-axis,the matric is:
             [1 0 0 0;0 0 -i 0;0 -i 0 0;0 0 0 1]
Argin:       qn_1        first qubit number
             qn_2        second qubit number
             error_rate  the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates :: iSWAP(size_t  qn_1,
                                size_t  qn_2,
                                double  error_rate,
                                Qnum    vControlBitNumber,
                                int     stQuantumBitNumber,
                                int     iDagger)
{
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        controlSwap
Description: iSWAP gate,both qubits swap and rotate π by z-axis,the matric is:
             [1 0 0 0;0 0 -i 0;0 -i 0 0;0 0 0 1]
Argin:       qn_1        first qubit number
             qn_2        second qubit number
             error_rate  the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::controlSwap(size_t qn_1, size_t qn_2, size_t qn_3, double error_rate)
{
    COMPLEX alpha, beta;
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t  ststep1 = (size_t)pow(2, qn_1);
        size_t  ststep2 = (size_t)pow(2, qn_2);
        size_t  ststep3 = (size_t)pow(2, qn_3 );

        /*
         *  traverse all the states
         */
        for (size_t i = ststep1; i < mvQuantumStat.size(); i = i + 2 * ststep1)
        {
            for (size_t j = i; j < i + ststep1; ++j)
            {
                if (j % (2 * ststep2) >= ststep2 && j % (2 * ststep3) < ststep3)
                {
                    alpha            = mvQuantumStat[j]; 
                    mvQuantumStat[j] = mvQuantumStat[j - ststep2 + ststep3];

                    mvQuantumStat[j - ststep2 + ststep3] = alpha;
                }
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        sqiSWAP
Description: sqiSWAP gate,both qubits swap and rotate π by z-axis,the matrix is:
             [1 0 0 0;0 1/sqrt(2) -i/sqrt(2) 0;0 -i/sqrt(2) 1/sqrt(2) 0;0 0 0 1]
Argin:       qn_1        first qubit number
             qn_2        second qubit number
             error_rate  the errorrate of the gate
             iDagger     is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::sqiSWAP(size_t  qn_1,
                                size_t  qn_2,
                                double  error_rate,
                                int     iDagger)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t sttemp, ststep1 = (size_t)pow(2, qn_1 ), ststep2 = (size_t)pow(2, qn_2 );
        /* 
         * sqiSWAP(qn_1,qn_2) is agree with
         * sqiSWAP(qn_2,qn_1) 
         */
        if (qn_1 < qn_2)                                                
        {
            sttemp = ststep1;
            ststep1 = ststep2;
            ststep2 = sttemp;
        }
        sttemp = ststep1 - ststep2;
        
        COMPLEX compll;
        
        if (!iDagger)
        {
            compll.real(0);
            compll.imag(-1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(1.0);
        }
        COMPLEX alpha, beta;
        
        /*
         *  traverse all the states
         */
        size_t j, k;
//#pragma omp parallel for private(j,k,alpha,beta)
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep1)
        {
            for (j = i + ststep2; j < i + ststep1; j = j + 2 * ststep2)
            {
                for (k = j; k < j + ststep2; k++)
                {
                    alpha = mvQuantumStat[k];
                    beta  = mvQuantumStat[k + sttemp];
                    
                    mvQuantumStat[k]          = alpha *SQ2 + compll *beta *SQ2;
                    mvQuantumStat[k + sttemp] = compll *alpha *SQ2 + beta *SQ2;
                }
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        sqiSWAP
Description: sqiSWAP gate,both qubits swap and rotate π by z-axis,the matrix is:
             [1 0 0 0;0 1/sqrt(2) -i/sqrt(2) 0;0 -i/sqrt(2) 1/sqrt(2) 0;0 0 0 1]
Argin:       qn_1               first qubit number
             qn_2               second qubit number
             error_rate         the errorrate of the gate
             vControlBitNumber  control bit number
             stQuantumBitNumber quantum bit number
             iDagger            is dagger
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates :: sqiSWAP(size_t  qn_1,
                                  size_t  qn_2,
                                  double  error_rate,
                                  Qnum    vControlBitNumber,
                                  int     stQuantumBitNumber,
                                  int     iDagger)
{
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        qubitMeasure
Description: measure qubit and the state collapsed
Argin:       qn    qubit number of the measurement
Argout:      None
return:      quantum error
*****************************************************************************************************************/
int X86QuantumGates::qubitMeasure(size_t qn)
{
    size_t ststep = (size_t)pow(2, qn);
    
    double dprob(0);

    for (size_t i = 0; i< mvQuantumStat.size(); i += ststep * 2)
    {
        for (size_t j = i; j<i + ststep; j++)
        {
            dprob += abs(mvQuantumStat[j])*abs(mvQuantumStat[j]);
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
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep)
        {
            for (j = i; j < i + ststep; j++)
            {
                mvQuantumStat[j]          *= dprob;
                mvQuantumStat[j + ststep]  = 0;
            }
        }
    }
    else
    {
        dprob = 1 / sqrt(1 - dprob); 

        size_t j;
//#pragma omp parallel for private(j)
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i = i + 2 * ststep)
        {
            for (j = i; j<i + ststep; j++) {
                mvQuantumStat[j]           = 0;
                mvQuantumStat[j + ststep] *= dprob;
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
QError X86QuantumGates::initState(QuantumGateParam *pQuantumProParam)
{
    if (nullptr == pQuantumProParam)
    {
        return undefineError;
    }

    size_t stQuantumStat = (size_t)pow(2, pQuantumProParam->mQuantumBitNumber);
    
    try
    {
        mvQuantumStat.resize(stQuantumStat);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return undefineError;
    }
    mvQuantumStat[0] = 1;

    return qErrorNone;
}


/*****************************************************************************************************************
Name:        matReverse
Description: change the position of the qubits in 2-qubit gate
Argin:       (*mat)[4]       pointer of the origin 2D array
             (*mat_rev)[4]   pointer of the changed 2D array
Argout:      2D array
return:      quantum error
*****************************************************************************************************************/
QError  X86QuantumGates::matReverse(COMPLEX(*mat)[4], COMPLEX(*mat_rev)[4])
{
    COMPLEX  temp;
    /*  mat_rev = { { mat[0][0],mat[0][2],mat[0][1],mat[0][3] },
     *            { mat[2][0],mat[2][2],mat[2][1],mat[2][3] },
     *            { mat[1][0],mat[1][2],mat[1][1],mat[1][3] },
     *            { mat[3][0],mat[3][2],mat[3][1],mat[3][3] }, };
     */
    for (size_t j = 0; j != 4; j++)                                     /* swap 2nd and 3rd row                 */
    {
        *(*mat_rev + j) = *(*mat + j);
        *(*(mat_rev + 1) + j) = *(*(mat + 2) + j);
        *(*(mat_rev + 2) + j) = *(*(mat + 1) + j);
        *(*(mat_rev + 3) + j) = *(*(mat + 3) + j);
    }

    for (size_t j = 0; j != 4; j++)                                     /* swap 2nd and 3rd column              */
    {
        temp = *(*(mat_rev + j) + 1);
        *(*(mat_rev + j) + 1) = *(*(mat_rev + j) + 2);
        *(*(mat_rev + j) + 2) = temp;
    }
    return qErrorNone;
}
/*****************************************************************************************************************
Name:        getCalculationUnitType
Description: compare calculation unit type
Argin:       sCalculationUnitType   external calculation unit type
Argout:      None
return:      comparison results
*****************************************************************************************************************/
bool X86QuantumGates :: compareCalculationUnitType(string& sCalculationUnitType)
{
    bool bResult = false;

    if (0 == sCalculationUnitType.compare(this->sCalculationUnitType))
    {
        bResult = true;
    }
    else
    {
        bResult = false;
    }

    return bResult;
}

/*****************************************************************************************************************
Name:        NOT
Description: NOT gate,invert the state.The matrix is
             [0 1;1 0]
Argin:       qn          qubit number that the Hadamard gate operates on.
             error_rate  the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::NOT(size_t qn, double error_rate)
{
    /*
     *  judge errorrate of the gate
     */
    if (randGenerator() > error_rate)
    {
        size_t ststep = (size_t)pow(2, qn);

        COMPLEX alpha, beta;

        size_t j;
//#pragma omp parallel for private(j,alpha,beta)
        for (LONG i = 0; i < (LONG)mvQuantumStat.size(); i += ststep * 2)
        {
            for (j = i; j<i + ststep; j++)
            {
                alpha = mvQuantumStat[j];
                beta  = mvQuantumStat[j + ststep];

                mvQuantumStat[j]          = beta;
                mvQuantumStat[j + ststep] = alpha;
            }
        }
    }

    return qErrorNone;
}

/*****************************************************************************************************************
Name:        NOT
Description: NOT gate,invert the state.The matrix is
[0 1;1 0]
Argin:       qn                 qubit number that the Hadamard gate operates on.
             error_rate         the errorrate of the gate
             vControlBit        control bit vector
             stQuantumBitNumber quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::NOT(size_t  qn,
                            double  error_rate,
                            Qnum    vControlBit,
                            int     stQuantumBitNumber)
{
    return qErrorNone;
}

/*****************************************************************************************************************
Name:        toffoli
Description: toffoli gate,the same as toffoli gate
Argin:       stControlBit1       first control qubit
             stControlBit2       the second control qubit
             stQuantumBit        target qubit
             errorRate           the errorrate of the gate
             stQuantumBitNumber  quantum bit number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates :: toffoli(size_t stControlBit1,
                                  size_t stControlBit2,
                                  size_t stQuantumBit,
                                  double errorRate,
                                  int    stQuantumBitNumber)
{
    if (randGenerator() > errorRate)
    {
		size_t ststep0 = 1ull << stControlBit1;
		size_t ststep1 = 1ull << stControlBit2;
		size_t ststep2 = 1ull << stQuantumBit;
        size_t temp;

        COMPLEX dtemp2(0);

        if (ststep0 > ststep1)
        {
            temp    = ststep1;
            ststep1 = ststep0;
            ststep0 = temp;
        }
        if (ststep0 > ststep2)
        {
            temp    = ststep2;
            ststep2 = ststep0;
            ststep0 = temp;
        }
        if (ststep1 > ststep2)
        {
            temp    = ststep2;
            ststep2 = ststep1;
            ststep1 = temp;
        }                                                               /* sort */

        temp = pow(2, stQuantumBit);
        size_t j,k,m;
//#pragma omp parallel for private(j,k,m)
        for (LONG i = ststep2; i < (LONG)mvQuantumStat.size(); i += 2 * ststep2)
        {
            for ( j = i + ststep1; j < i + ststep2; j += 2 * ststep1)
            {
                for ( k = j + ststep0; k < j + ststep1; k += 2 * ststep0)
                {
                    for ( m = k; m < k + ststep0; m++)
                    {
                        dtemp2           = mvQuantumStat[m];
                        mvQuantumStat[m] = mvQuantumStat[m - temp];

                        mvQuantumStat[m - temp] = dtemp2;
                    }
                }
            }
        }

    }
    return qErrorNone;

}

/*****************************************************************************************************************
Name:        endGate
Description: end gate
Argin:       pQuantumProParam       quantum program param pointer
             pQGate                 quantum gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError X86QuantumGates::endGate(QuantumGateParam * pQuantumProParam, QuantumGates * pQGate)
{

    for (size_t i = 0; i < mvQuantumStat.size(); i++)
    {
        mvQuantumStat[i] = 0;
    }
    mvQuantumStat[0] =1;
    return qErrorNone;
}

/*************************************************
Name:      unitarySingleQubitGate
Description:        unitary single qubit gate
Argin:            psi        reference of vector<complex<double>> which contains quantum state information.
qn        target qubit number
matrix  matrix of the gate
error_rate  the errorrate of the gate
Argout:    the state after measurement
return:    None
*************************************************/
void X86QuantumGates::unitarySingleQubitGate(size_t qn, QStat& matrix, double error_rate)
{
    /*
    *  judge errorrate of the gate
    */
    if (randGenerator() > error_rate)
    {
        size_t ststep = pow(2, qn);
        COMPLEX alpha, beta;
        for (size_t i = 0; i< mvQuantumStat.size(); i += ststep * 2)
        {
            for (size_t j = i; j<i + ststep; j++)
            {
                alpha = mvQuantumStat[j];
                beta = mvQuantumStat[j + ststep];
                mvQuantumStat[j] = alpha * matrix[0] + beta * matrix[1];              /* in j,the goal qubit is in |0>        */
                mvQuantumStat[j + ststep] = alpha * matrix[2] + beta * matrix[3];        /* in j+ststep,the goal qubit is in |1> */
            }
        }
    }
}

/*************************************************
Name:      unitarySingleQubitGateDagger
Description:        unitary single qubit gate
Argin:            psi        reference of vector<complex<double>> which contains quantum state information.
qn        target qubit number
matrix  matrix of the gate
error_rate  the errorrate of the gate
Argout:    the state after measurement
return:    None
*************************************************/
void X86QuantumGates::unitarySingleQubitGateDagger(size_t qn, QStat& matrix, double error_rate)
{
    /*
    *  judge errorrate of the gate
    */
    if (randGenerator() > error_rate)
    {
        size_t ststep = pow(2, qn);
        QStat matrixdagger(4);
        for (size_t i = 0; i < 4; i++)
        {
            matrix[i] = COMPLEX(matrix[i].real(), -matrix[i].imag());
        }
        COMPLEX temp;
        temp = matrixdagger[1];
        matrixdagger[1] = matrixdagger[2];
        matrixdagger[2] = temp;
        COMPLEX alpha, beta;
        for (size_t i = 0; i< mvQuantumStat.size(); i += ststep * 2)
        {
            for (size_t j = i; j<i + ststep; j++)
            {
                alpha = mvQuantumStat[j];
                beta = mvQuantumStat[j + ststep];
                mvQuantumStat[j] = alpha * matrix[0] + beta * matrix[1];              /* in j,the goal qubit is in |0>        */
                mvQuantumStat[j + ststep] = alpha * matrix[2] + beta * matrix[3];        /* in j+ststep,the goal qubit is in |1> */
            }
        }
    }
}

void X86QuantumGates::controlunitarySingleQubitGate(Qnum& qnum, QStat& matrix, double error_rate)
{
    if (randGenerator() > error_rate)
    {
        size_t qn = qnum.back();                                        /*qn is target number*/
        sort(qnum.begin(), qnum.end());
        int M = mvQuantumStat.size() / pow(2, qnum.size()), x;
        int n = log(mvQuantumStat.size()) / log(2);
        size_t ststep = pow(2, qn);
        COMPLEX alpha, beta;
        size_t index = 0, block = 0;
        for (Qnum::iterator j = qnum.begin(); j != qnum.end(); j++)
        {
            block += pow(2, *j);
        }
        Qnum::iterator qiter;
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qnum.begin();
            for (int j = 0; j < n; j++)
            {
                while (qiter != qnum.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (x % 2)*pow(2, j);
                x >>= 1;
            }
            index = index + block - ststep;                             /*control qubits are 1,target qubit is 0 */
            //cout << index << endl;
            alpha = mvQuantumStat[index];
            beta = mvQuantumStat[index + ststep];
            mvQuantumStat[index] = alpha * matrix[0] + beta * matrix[1];
            mvQuantumStat[index + ststep] = alpha * matrix[2] + beta * matrix[3];
        }
    }
}

void X86QuantumGates::controlunitarySingleQubitGateDagger(Qnum& qnum, QStat& matrix, double error_rate)
{
    if (randGenerator() > error_rate)
    {
        COMPLEX temp;
        temp = matrix[1];
        matrix[1] = matrix[2];
        matrix[2] = temp;  //转置
        for (size_t i = 0; i < 3; i++)
        {
            matrix[i] = COMPLEX(matrix[i].real(), -matrix[i].imag());
        }//共轭
        size_t qn = qnum.back();                                        /*qn is target number                   */
        sort(qnum.begin(), qnum.end());
        int M = mvQuantumStat.size() / pow(2, qnum.size()), x;
        int n = log(mvQuantumStat.size()) / log(2);
        size_t ststep = pow(2, qn);
        COMPLEX alpha, beta;
        size_t index = 0, block = 0;
        for (Qnum::iterator j = qnum.begin(); j != qnum.end(); j++)
        {
            block += pow(2, *j);
        }

        Qnum::iterator qiter;
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qnum.begin();
            for (int j = 0; j < n; j++)        //n qubits
            {
                while (qiter != qnum.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (x % 2)*pow(2, j);
                x >>= 1;
            }
            index = index + block - ststep;                             /*control qubits are 1,target qubit is 0 */
            //cout << index << endl;
            alpha = mvQuantumStat[index];
            beta = mvQuantumStat[index + ststep];
            mvQuantumStat[index] = alpha * matrix[0] + beta * matrix[1];
            mvQuantumStat[index + ststep] = alpha * matrix[2] + beta * matrix[3];
        }
    }
}







void X86QuantumGates::unitaryDoubleQubitGate(size_t qn_1, size_t qn_2, QStat& matrix, double error_rate)
{
    /*
    *  judge errorrate of the gate
    */
    if (randGenerator() > error_rate)
    {
        size_t ststep1 = pow(2, qn_1), ststep2 = pow(2, qn_2);
        COMPLEX phi00, phi01, phi10, phi11;
        if (qn_1>qn_2)                                                  /* control qubit number is higher       */
        {
            /*
            *  traverse all the states
            */
            for (size_t i = 0; i<mvQuantumStat.size(); i = i + 2 * ststep1)
            {
                for (size_t j = i; j < i + ststep1; j = j + 2 * ststep2)
                {
                    for (size_t k = j; k < j + ststep2; k++)
                    {
                        phi00 = mvQuantumStat[k];        //00
                        phi01 = mvQuantumStat[k + ststep2];  //01
                        phi10 = mvQuantumStat[k + ststep1];  //10
                        phi11 = mvQuantumStat[k + ststep1 + ststep2]; //11
                        mvQuantumStat[k] = matrix[0] * phi00 + matrix[1] * phi01
                            + matrix[2] * phi10 + matrix[3] * phi11;
                        mvQuantumStat[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                            + matrix[6] * phi10 + matrix[7] * phi11;
                        mvQuantumStat[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                            + matrix[10] * phi10 + matrix[11] * phi11;
                        mvQuantumStat[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                            + matrix[14] * phi10 + matrix[15] * phi11;

                    }
                }
            }
        }
        else                                                            /* control qubit numer is lower         */
        {
            for (size_t i = 0; i<mvQuantumStat.size(); i = i + 2 * ststep2)
            {
                for (size_t j = i; j < i + ststep2; j = j + 2 * ststep1)
                {
                    for (size_t k = j; k < j + ststep1; k++)
                    {
                        phi00 = mvQuantumStat[k];
                        phi01 = mvQuantumStat[k + ststep2];
                        phi10 = mvQuantumStat[k + ststep1];
                        phi11 = mvQuantumStat[k + ststep1 + ststep2];
                        mvQuantumStat[k] = matrix[0] * phi00 + matrix[1] * phi01
                            + matrix[2] * phi10 + matrix[3] * phi11;
                        mvQuantumStat[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                            + matrix[6] * phi10 + matrix[7] * phi11;
                        mvQuantumStat[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                            + matrix[10] * phi10 + matrix[11] * phi11;
                        mvQuantumStat[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                            + matrix[14] * phi10 + matrix[15] * phi11;
                    }
                }
            }
        }
    }
}


void X86QuantumGates::unitaryDoubleQubitGateDagger(size_t qn_1, size_t qn_2, QStat& matrix, double error_rate)
{
    /*
    *  judge errorrate of the gate
    */
    if (randGenerator() > error_rate)
    {
        size_t ststep1 = pow(2, qn_1), ststep2 = pow(2, qn_2);
        COMPLEX temp, phi00, phi01, phi10, phi11;
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = i + 1; j < 4; j++)
            {
                temp = matrix[4 * i + j];
                matrix[4 * i + j] = matrix[4 * j + i];
                matrix[4 * j + i] = temp;
            }
        }//转置
        for (size_t i = 0; i < 16; i++)
        {
            //matrix[i].imag = -matrix[i].imag;
            matrix[i] = COMPLEX(matrix[i].real(), -matrix[i].imag());
        }//共轭
        if (qn_1>qn_2)                                                  /* control qubit number is higher       */
        {
            /*
            *  traverse all the states
            */
            for (size_t i = 0; i<mvQuantumStat.size(); i = i + 2 * ststep1)
            {
                for (size_t j = i; j < i + ststep1; j = j + 2 * ststep2)
                {
                    for (size_t k = j; k < j + ststep2; k++)
                    {
                        phi00 = mvQuantumStat[k];
                        phi01 = mvQuantumStat[k + ststep2];
                        phi10 = mvQuantumStat[k + ststep1];
                        phi11 = mvQuantumStat[k + ststep1 + ststep2];
                        mvQuantumStat[k] = matrix[0] * phi00 + matrix[1] * phi01
                            + matrix[2] * phi10 + matrix[3] * phi11;
                        mvQuantumStat[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                            + matrix[6] * phi10 + matrix[7] * phi11;
                        mvQuantumStat[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                            + matrix[10] * phi10 + matrix[11] * phi11;
                        mvQuantumStat[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                            + matrix[14] * phi10 + matrix[15] * phi11;

                    }
                }
            }
        }
        else                                                            /* control qubit numer is lower         */
        {
            for (size_t i = 0; i<mvQuantumStat.size(); i = i + 2 * ststep2)
            {
                for (size_t j = i; j < i + ststep2; j = j + 2 * ststep1)
                {
                    for (size_t k = j; k < j + ststep1; k++)
                    {
                        phi00 = mvQuantumStat[k];
                        phi01 = mvQuantumStat[k + ststep2];
                        phi10 = mvQuantumStat[k + ststep1];
                        phi11 = mvQuantumStat[k + ststep1 + ststep2];
                        mvQuantumStat[k] = matrix[0] * phi00 + matrix[1] * phi01
                            + matrix[2] * phi10 + matrix[3] * phi11;
                        mvQuantumStat[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                            + matrix[6] * phi10 + matrix[7] * phi11;
                        mvQuantumStat[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                            + matrix[10] * phi10 + matrix[11] * phi11;
                        mvQuantumStat[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                            + matrix[14] * phi10 + matrix[15] * phi11;
                    }
                }
            }
        }
    }
}


void X86QuantumGates::controlunitaryDoubleQubitGate(Qnum& qnum, QStat& matrix, double error_rate)
{
    if (randGenerator() > error_rate)
    {
        size_t qn_1 = qnum[qnum.size() - 2];
        size_t qn_2 = qnum[qnum.size() - 1];  //qnum最后两个元素表示双门作用的两个比特
        sort(qnum.begin(), qnum.end());
        int M = mvQuantumStat.size() / pow(2, qnum.size()), x;
        int n = log(mvQuantumStat.size()) / log(2);
        size_t ststep1 = pow(2, qn_1);
        size_t ststep2 = pow(2, qn_2);
        COMPLEX phi00, phi01, phi10, phi11;
        size_t index = 0, block = 0;
        for (Qnum::iterator j = qnum.begin(); j != qnum.end(); j++)
        {
            block += pow(2, *j);
        }

        Qnum::iterator qiter;
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qnum.begin();
            for (int j = 0; j < n; j++)
            {
                while (qiter != qnum.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (x % 2)*pow(2, j);
                x >>= 1;
            }
            index = index + block - ststep1 - ststep2;                             /*control qubits are 1,target qubit are 0 */
            phi00 = mvQuantumStat[index];
            phi01 = mvQuantumStat[index + ststep2];
            phi10 = mvQuantumStat[index + ststep1];
            phi11 = mvQuantumStat[index + ststep1 + ststep2];
            mvQuantumStat[index] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            mvQuantumStat[index + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            mvQuantumStat[index + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            mvQuantumStat[index + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
}

void X86QuantumGates::controlunitaryDoubleQubitGateDagger(Qnum& qnum, QStat& matrix, double error_rate)
{
    if (randGenerator() > error_rate)
    {
        COMPLEX temp;
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = i + 1; j < 4; j++)
            {
                temp = matrix[4 * i + j];
                matrix[4 * i + j] = matrix[4 * j + i];
                matrix[4 * j + i] = temp;
            }
        }//转置
        for (size_t i = 0; i < 16; i++)
        {
            matrix[i] = COMPLEX(matrix[i].real(), -matrix[i].imag());
        }//共轭
        size_t qn_1 = qnum[qnum.size() - 2];
        size_t qn_2 = qnum[qnum.size() - 1];  //qnum最后两个元素表示双门作用的两个比特
        sort(qnum.begin(), qnum.end());
        int M = mvQuantumStat.size() / pow(2, qnum.size()), x;
        int n = log(mvQuantumStat.size()) / log(2);
        size_t ststep1 = pow(2, qn_1);
        size_t ststep2 = pow(2, qn_2);
        COMPLEX phi00, phi01, phi10, phi11;
        size_t index = 0, block = 0;
        for (Qnum::iterator j = qnum.begin(); j != qnum.end(); j++)
        {
            block += pow(2, *j);
        }

        Qnum::iterator qiter;
        for (size_t i = 0; i < M; i++)
        {
            index = 0;
            x = i;
            qiter = qnum.begin();
            for (int j = 0; j < n; j++)
            {
                while (qiter != qnum.end() && *qiter == j)
                {
                    qiter++;
                    j++;
                }
                index += (x % 2)*pow(2, j);
                x >>= 1;
            }
            index = index + block - ststep1 - ststep2;                             /*control qubits are 1,target qubit are 0 */
            phi00 = mvQuantumStat[index];
            phi01 = mvQuantumStat[index + ststep2];
            phi10 = mvQuantumStat[index + ststep1];
            phi11 = mvQuantumStat[index + ststep1 + ststep2];
            mvQuantumStat[index] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            mvQuantumStat[index + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            mvQuantumStat[index + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            mvQuantumStat[index + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
}
