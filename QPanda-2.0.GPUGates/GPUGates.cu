/******************************************************************************
Copyright (c) 2017-2018 Origin Quantum Computing Co., Ltd.. All Rights Reserved.


 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 

 	http://www.apache.org/licenses/LICENSE-2.0 
 
Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language
governing permissions and 
limitations under the License.

Author:Xue Cheng
Date:2017-12-13
Description: Definition of Cuda function of gates
************************************************************************/
#include "GPUGatesDecl.h"
#include <vector>
#include <algorithm>
#include <time.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define __cplusplus
#define __CUDACC__
#include < device_functions.h>
using namespace std;	
#define QSIZE   size_t

#define SQ2 0.707106781186548
#define PI 3.141592653589793
#define THREADDIM 1024

namespace gpu {
//typedef quantumstate QState;
typedef std::vector<STATE_T> vecdou;
//typedef std::vector<probability> vecprob;
__global__ void unitarysingle(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T *matrix_real,
    STATE_T *matrix_imag);

__global__ void controlunitarysingle(
    STATE_T * psireal, 
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE Block2,
    QSIZE Block3, 
    STATE_T *matrix_real,
    STATE_T *matrix_imag );


__global__ void unitarydouble(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    STATE_T *matrix,
    QSIZE Block1,
    QSIZE Block2);
/****************************************
Dim:1<<qnum
block1: involved qubits array
block2:first qubit
block3:second qubit
block4:involved qubit number
matrix_real:real component of matrix,[16,1]
matrix_imag:imag component of matrix,[16,1]
************************************************/
__global__ void controlunitarydouble(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE Block2,
    QSIZE Block3,
    QSIZE Block4,
    STATE_T *matrix_real,
    STATE_T *matrix_imag);



__global__ void initState(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim);
__global__ void Hadamard(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block);
__global__ void Hadamardnew(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block);
__global__ void controlHadamard(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m);

__global__ void X(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block);
__global__ void controlX(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m);

__global__ void Y(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block);
__global__ void controlY(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m);

__global__ void Z(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block);
__global__ void controlZ(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m);

__global__ void S(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, int ilabel);
__global__ void controlS(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m, 
    int ilabel);

__global__ void T(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, int ilabel);
__global__ void controlT(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2, 
    QSIZE m,
    int ilabel);



__global__ void RX(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void controlRX(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim, 
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void RY(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void controlRY(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void RZ(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void controlRZ(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim, 
    QSIZE * block1,
    QSIZE  block2, 
    QSIZE m, 
    STATE_T costheta,
    STATE_T sintheta);

__global__ void CNOT(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block1, QSIZE Block2);
__global__ void CZ(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block1, QSIZE Block2);
__global__ void CR(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block1,
    QSIZE Block2,
    STATE_T costheta,
    STATE_T sintheta);
__global__ void iSWAP(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block1,
    QSIZE Block2,
    STATE_T costheta,
    STATE_T sintheta);

__global__ void controliSWAP(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block,
    QSIZE Block1,
    QSIZE Block2,
    QSIZE m,
    STATE_T costheta,
    STATE_T sintheta);




__global__ void qubitprob(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T *pr);
__global__ void probsum(STATE_T * pr, STATE_T * prob);
__global__ void qubitcollapse0(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T coef);
__global__ void qubitcollapse1(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T coef);


__global__ void multiprob(
    STATE_T * psireal,
    STATE_T * psiimag, 
    QSIZE Dim,
    STATE_T * pro,
    QSIZE * block,
    QSIZE m,
    QSIZE dec);

__global__ void probsumnew1(STATE_T * psireal, STATE_T * psiimag, STATE_T *probtemp, size_t num1, size_t m, size_t Dim, size_t * block);


double randGenerator();

/*
bool RX(QState& psi, size_t, double, double error_rate = 0);
bool RXdagger(QState& psi, size_t, double, double error_rate = 0);
bool RY(QState& psi, size_t, double, double error_rate = 0);
bool RYdagger(QState& psi, size_t, double, double error_rate = 0);
bool RZ(QState& psi, size_t, double, double error_rate = 0);
bool RZdagger(QState& psi, size_t, double, double error_rate = 0);
bool NOT(QState& psi, size_t qn, double error_rate = 0);
bool NOTdagger(QState& psi, size_t qn, double error_rate = 0);
bool CNOT(QState& psi, size_t, size_t, double error_rate = 0);
bool CNOTdagger(QState& psi, size_t, size_t, double error_rate = 0);
bool CR(QState& psi, size_t, size_t, double, double error_rate = 0);
bool CRdagger(QState& psi, size_t, size_t, double, double error_rate = 0);
bool iSWAP(QState& psi, size_t, size_t, double error_rate = 0);
bool iSWAPdagger(QState& psi, size_t, size_t, double error_rate = 0);
bool sqiSWAP(QState& psi, size_t, size_t, double error_rate = 0);
bool sqiSWAPdagger(QState& psi, size_t, size_t, double error_rate = 0);
int qubitmeasure(QState& psi, QSIZE Block);
bool controlHadamard(QState& psi, Qnum&, double error_rate = 0);
bool controlHadamarddagger(QState& psi, Qnum&, double error_rate = 0);
bool controlRX(QState& psi, Qnum&, double, double error_rate = 0);
bool controlRXdagger(QState& psi, Qnum&, double, double error_rate = 0);
bool controlRY(QState& psi, Qnum&, double, double error_rate = 0);
bool controlRYdagger(QState& psi, Qnum&, double, double error_rate = 0);
bool controlRZ(QState& psi, Qnum&, double, double error_rate = 0);
bool controlRZdagger(QState& psi, Qnum&, double, double error_rate = 0);
// bool toffoli(QState& psi, size_t, size_t, size_t, double error_rate = 0);
// bool toffolidagger(QState& psi, size_t, size_t, size_t, double error_rate = 0);
bool qbReset(QState& psi, size_t, double error_rate = 0);
bool pMeasure(QState&, vecprob&, QSIZE *block, QSIZE m);
bool pMeasurenew(QState&, vector<pair<size_t, double>>&, Qnum&);
bool getState(QState &psi,QState &psigpu,int qnum);
double randGenerator();
*/


/***************************************************************************************
Probdis pMeasure(QState&, Qnum&);
********************************************************************************************/





__global__ void 
unitarysingle(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block,
    STATE_T *matrix_real,
    STATE_T *matrix_imag)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = matrix_real[0]*X1- matrix_imag[0]*Y1+ matrix_real[1] * X2 - matrix_imag[1] * Y2;
        psireal[corIdx] = matrix_real[2] * X1 - matrix_imag[2] * Y1 + matrix_real[3] * X2 - matrix_imag[3] * Y2;
        psiimag[realIdx] = matrix_real[0] * Y1 + matrix_imag[0] * X1 + matrix_real[1] * Y2 + matrix_imag[1] * X2;
        psiimag[corIdx] = matrix_real[2] * Y1 + matrix_imag[2] * X1 + matrix_real[3] * Y2 + matrix_imag[3] * X2;
    }
}



__global__ void controlunitarysingle(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE  block2,
    QSIZE m,
    STATE_T *matrix_real,
    STATE_T *matrix_imag
    )
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = matrix_real[0]*X1- matrix_imag[0]*Y1+ matrix_real[1] * X2 - matrix_imag[1] * Y2;
        psireal[corIdx] = matrix_real[2] * X1 - matrix_imag[2] * Y1 + matrix_real[3] * X2 - matrix_imag[3] * Y2;
        psiimag[realIdx] = matrix_real[0] * Y1 + matrix_imag[0] * X1 + matrix_real[1] * Y2 + matrix_imag[1] * X2;
        psiimag[corIdx] = matrix_real[2] * Y1 + matrix_imag[2] * X1 + matrix_real[3] * Y2 + matrix_imag[3] * X2;
    }

}


__global__ void 
unitarydouble(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block1,
    QSIZE Block2,
    STATE_T *matrix_real,
    STATE_T *matrix_imag)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
    QSIZE Idx00, Idx01, Idx10, Idx11;
    if (Block1 >  Block2)
    {
        Idx10 = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + idx%  Block2;
    }
    else
    {
        Idx10 = (idx / (Block2 / 2)) * 2 * Block2 + (idx % (Block2 / 2) / Block1) * 2 * Block1 + Block1 + idx%  Block1;
    }
    Idx00 = Idx10 - Block1;
    Idx01 = Idx00 + Block2;
    Idx11 = Idx10 + Block2;
    
    if (Idx11 < Dim)
    {
        STATE_T X00 = psireal[Idx00];
        STATE_T X01 = psireal[Idx01];
        STATE_T X10 = psireal[Idx10];
        STATE_T X11 = psireal[Idx11];
        STATE_T Y00 = psiimag[Idx00];
        STATE_T Y01 = psiimag[Idx01];
        STATE_T Y10 = psiimag[Idx10];
        STATE_T Y11 = psiimag[Idx11];
        psireal[Idx00] = matrix_real[0] * X00 - matrix_imag[0] * Y00
            + matrix_real[1] * X01 - matrix_imag[1] * Y01
            + matrix_real[2] * X10 - matrix_imag[2] * Y10
            + matrix_real[3] * X11 - matrix_imag[3] * Y11;
        psiimag[Idx00] = matrix_imag[0] * X00 + matrix_real[0] * Y00
            + matrix_imag[1] * X01 + matrix_real[1] * Y01
            + matrix_imag[2] * X10 + matrix_real[2] * Y10
            + matrix_imag[3] * X11 + matrix_real[3] * Y11;

        psireal[Idx01] = matrix_real[4] * X00 - matrix_imag[4] * Y00
            + matrix_real[5] * X01 - matrix_imag[5] * Y01
            + matrix_real[6] * X10 - matrix_imag[6] * Y10
            + matrix_real[7] * X11 - matrix_imag[7] * Y11;
        psiimag[Idx01] = matrix_imag[4] * X00 + matrix_real[4] * Y00
            + matrix_imag[5] * X01 + matrix_real[5] * Y01
            + matrix_imag[6] * X10 + matrix_real[6] * Y10
            + matrix_imag[7] * X11 + matrix_real[7] * Y11;

        psireal[Idx10] = matrix_real[8] * X00 - matrix_imag[8] * Y00
            + matrix_real[9] * X01 - matrix_imag[9] * Y01
            + matrix_real[10] * X10 - matrix_imag[10] * Y10
            + matrix_real[11] * X11 - matrix_imag[11] * Y11;
        psiimag[Idx10] = matrix_imag[8] * X00 + matrix_real[8] * Y00
            + matrix_imag[9] * X01 + matrix_real[9] * Y01
            + matrix_imag[10] * X10 + matrix_real[10] * Y10
            + matrix_imag[11] * X11 + matrix_real[11] * Y11;
        psireal[Idx11] = matrix_real[12] * X00 - matrix_imag[12] * Y00
            + matrix_real[13] * X01 - matrix_imag[13] * Y01
            + matrix_real[14] * X10 - matrix_imag[14] * Y10
            + matrix_real[15] * X11 - matrix_imag[15] * Y11;
        psiimag[Idx11] = matrix_imag[12] * X00 + matrix_real[12] * Y00
            + matrix_imag[13] * X01 + matrix_real[13] * Y01
            + matrix_imag[14] * X10 + matrix_real[14] * Y10
            + matrix_imag[15] * X11 + matrix_real[15] * Y11;
    }
}


__global__ void controlunitarydouble(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block1,
    QSIZE Block2,
    QSIZE Block3,
    QSIZE Block4,
    STATE_T *matrix_real,
    STATE_T *matrix_imag)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE Idx00, Idx01, Idx10, Idx11;
    QSIZE i, j, k;
	QSIZE index;
    if (idx < Dim / (1 << Block4))
    {
        QSIZE index = idx;
        for (i = 0; i < Block4; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;
        }
    }
    
    Idx00 = index - Block2-Block3;
    Idx01 = Idx00 - Block2;
    Idx10 = Idx10 - Block3;
    Idx11 = index;

    if (Idx11 < Dim)
    {
        STATE_T X00 = psireal[Idx00];
        STATE_T X01 = psireal[Idx01];
        STATE_T X10 = psireal[Idx10];
        STATE_T X11 = psireal[Idx11];
        STATE_T Y00 = psiimag[Idx00];
        STATE_T Y01 = psiimag[Idx01];
        STATE_T Y10 = psiimag[Idx10];
        STATE_T Y11 = psiimag[Idx11];
        psireal[Idx00] = matrix_real[0] * X00 - matrix_imag[0] * Y00
            + matrix_real[1] * X01 - matrix_imag[1] * Y01
            + matrix_real[2] * X10 - matrix_imag[2] * Y10
            + matrix_real[3] * X11 - matrix_imag[3] * Y11;
        psiimag[Idx00] = matrix_imag[0] * X00 + matrix_real[0] * Y00
            + matrix_imag[1] * X01 + matrix_real[1] * Y01
            + matrix_imag[2] * X10 + matrix_real[2] * Y10
            + matrix_imag[3] * X11 + matrix_real[3] * Y11;

        psireal[Idx01] = matrix_real[4] * X00 - matrix_imag[4] * Y00
            + matrix_real[5] * X01 - matrix_imag[5] * Y01
            + matrix_real[6] * X10 - matrix_imag[6] * Y10
            + matrix_real[7] * X11 - matrix_imag[7] * Y11;
        psiimag[Idx01] = matrix_imag[4] * X00 + matrix_real[4] * Y00
            + matrix_imag[5] * X01 + matrix_real[5] * Y01
            + matrix_imag[6] * X10 + matrix_real[6] * Y10
            + matrix_imag[7] * X11 + matrix_real[7] * Y11;

        psireal[Idx10] = matrix_real[8] * X00 - matrix_imag[8] * Y00
            + matrix_real[9] * X01 - matrix_imag[9] * Y01
            + matrix_real[10] * X10 - matrix_imag[10] * Y10
            + matrix_real[11] * X11 - matrix_imag[11] * Y11;
        psiimag[Idx10] = matrix_imag[8] * X00 + matrix_real[8] * Y00
            + matrix_imag[9] * X01 + matrix_real[9] * Y01
            + matrix_imag[10] * X10 + matrix_real[10] * Y10
            + matrix_imag[11] * X11 + matrix_real[11] * Y11;
        psireal[Idx11] = matrix_real[12] * X00 - matrix_imag[12] * Y00
            + matrix_real[13] * X01 - matrix_imag[13] * Y01
            + matrix_real[14] * X10 - matrix_imag[14] * Y10
            + matrix_real[15] * X11 - matrix_imag[15] * Y11;
        psiimag[Idx11] = matrix_imag[12] * X00 + matrix_real[12] * Y00
            + matrix_imag[13] * X01 + matrix_real[13] * Y01
            + matrix_imag[14] * X10 + matrix_real[14] * Y10
            + matrix_imag[15] * X11 + matrix_real[15] * Y11;
    }

}








__global__ void  initState(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    if (idx < Dim / 2 && idx != 0)
    {
        psireal[idx] = 0;
        psiimag[idx] = 0;
        psireal[idx + Dim / 2] = 0;
        psiimag[idx + Dim / 2] = 0;
    }
    if (0 == idx)
    {
        psireal[0] = 1;
        psiimag[0] = 0;
        psireal[Dim / 2] = 0;
        psiimag[Dim / 2] = 0;
    }
}

__global__ void  Hadamardnew(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE realIdx;
    QSIZE corIdx;
    STATE_T X1, X2, Y1, Y2;

     for (int i = idx; i < Dim; i += gridDim.x*blockDim.x)
    {
         realIdx = i / (Block<<1) * 2 * Block + i%Block ;
         corIdx = realIdx + Block;
          X1 = psireal[realIdx];
          X2 = psireal[corIdx];
          Y1 = psiimag[realIdx];
          Y2 = psiimag[corIdx];
         psireal[realIdx] = (X1 + X2)*SQ2;
         psireal[corIdx] = (X1 - X2)*SQ2;
         psiimag[realIdx] = (Y1 + Y2)*SQ2;
         psiimag[corIdx] = (Y1 - Y2)*SQ2;

     }
    
}

//Hadamard
__global__ void Hadamard(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = (X1 + X2)*SQ2;
        psireal[corIdx] = (X1 - X2)*SQ2;
        psiimag[realIdx] = (Y1 + Y2)*SQ2;
        psiimag[corIdx] = (Y1 - Y2)*SQ2;
    }
}

__global__ void controlHadamard(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = (X1 + X2)*SQ2;
        psireal[corIdx] = (X1 - X2)*SQ2;
        psiimag[realIdx] = (Y1 + Y2)*SQ2;
        psiimag[corIdx] = (Y1 - Y2)*SQ2;
    }
}//checked

//X gate
__global__ void X(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X2;
        psireal[corIdx] = X1;
        psiimag[realIdx] = Y2;
        psiimag[corIdx] = Y1;
    }
}

__global__ void controlX(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X2;
        psireal[corIdx] = X1;
        psiimag[realIdx] = Y2;
        psiimag[corIdx] = Y1;
    }
}


 //Y gate
__global__ void Y(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = Y2;
        psireal[corIdx] = Y1;
        psiimag[realIdx] = -X2;
        psiimag[corIdx] = -X1;
    }
}

__global__ void controlY(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = Y2;
        psireal[corIdx] = Y1;
        psiimag[realIdx] = -X2;
        psiimag[corIdx] = -X1;
    }
}

//Z gate
__global__ void Z(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        psireal[corIdx] = -psireal[corIdx];
        psiimag[corIdx] = -psiimag[corIdx];
    }
}

__global__ void controlZ(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        psireal[corIdx] = -psireal[corIdx];
        psiimag[corIdx] = -psiimag[corIdx];
    }
}

//S gate
__global__ void S(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block,int ilabel)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        psireal[corIdx] = -psiimag[corIdx]*ilabel;
        psiimag[corIdx] = psireal[corIdx]*ilabel;
    }
}

__global__ void controlS(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m,int ilabel)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        psireal[corIdx] = -psiimag[corIdx] * ilabel;
        psiimag[corIdx] = psireal[corIdx] * ilabel;
    }
}

//T gate
__global__ void T(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, int ilabel)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[corIdx];
        STATE_T Y1 = psiimag[corIdx];
        psireal[corIdx] = (X1-Y1*ilabel)*SQ2;
        psiimag[corIdx] = (X1 + Y1 * ilabel)*SQ2;
    }
}

__global__ void controlT(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m, int ilabel)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        STATE_T X1 = psireal[corIdx];
        STATE_T Y1 = psiimag[corIdx];
        psireal[corIdx] = (X1 - Y1 * ilabel)*SQ2;
        psiimag[corIdx] = (X1 + Y1 * ilabel)*SQ2;
    }
}
//RX
__global__ void RX(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx % Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;
    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X1 * costheta + Y2 * sintheta;
        psireal[corIdx] = Y1 * sintheta + X2 * costheta;
        psiimag[realIdx] = Y1 * costheta - X2 * sintheta;
        psiimag[corIdx] = Y2 * costheta - X1 * sintheta;
    }
}
__global__ void controlRX(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X1 * costheta + Y2 * sintheta;
        psireal[corIdx] = Y1 * sintheta + X2 * costheta;
        psiimag[realIdx] = Y1 * costheta - X2 * sintheta;
        psiimag[corIdx] = Y2 * costheta - X1 * sintheta;
    }
}

//RY

__global__ void  RY(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;
    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X1*costheta - X2*sintheta;
        psireal[corIdx] = X1*sintheta + X2*costheta;
        psiimag[realIdx] = Y1*costheta - Y2*sintheta;
        psiimag[corIdx] = Y2*costheta + Y1*sintheta;
    }
}

__global__ void controlRY(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

                                                               //QSIZE BlockNum = idx / Block;
                                                               //QSIZE BlockInt = idx% Block;    
    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111
        QSIZE realIdx = corIdx - block2;                    //1110
        STATE_T X1 = psireal[realIdx];
        STATE_T X2 = psireal[corIdx];
        STATE_T Y1 = psiimag[realIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = X1 * costheta - X2 * sintheta;
        psireal[corIdx] = X1 * sintheta + X2 * costheta;
        psiimag[realIdx] = Y1 * costheta - Y2 * sintheta;
        psiimag[corIdx] = Y2 * costheta + Y1 * sintheta;
    }
}


//RZ
__global__ void  RZ(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;

    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE corIdx = BlockNum * 2 * Block + BlockInt + Block;
    if (corIdx < Dim)
    {
        STATE_T X2 = psireal[corIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[corIdx] = X2*costheta - Y2*sintheta;
        psiimag[corIdx] = X2*sintheta + Y2*costheta;
    }
}
__global__ void controlRZ(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE * block1, QSIZE  block2, QSIZE m, STATE_T costheta, STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block1[i];
            k = index % block1[i];
            index = j * 2 * block1[i] + block1[i] + k;

        }
        QSIZE corIdx = index;                                   //1111        
        STATE_T X2 = psireal[corIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[corIdx] = X2 * costheta - Y2 * sintheta;
        psiimag[corIdx] = X2 * sintheta + Y2 * costheta;
    }
}


//CNOT
__global__ void  CNOT(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block1, QSIZE Block2)    //2^(qnum)           q9q8q7...q0 
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
    QSIZE corIdx, realIdx;
    if (Block1 >  Block2)
    {
        corIdx = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + idx%  Block2;
    }
    else
    {
        corIdx = (idx / (Block2 / 2)) * 2 * Block2 + (idx % (Block2 / 2) / Block1) * 2 * Block1 + Block1 + idx%  Block1;
    }
    realIdx = corIdx + Block2;
    if (realIdx < Dim)
    {
        STATE_T X1 = psireal[corIdx];                                   //10
        STATE_T Y1 = psiimag[corIdx];
        STATE_T X2 = psireal[realIdx];                                  //11
        STATE_T Y2 = psiimag[realIdx];
        psireal[corIdx] = X2;
        psiimag[corIdx] = Y2;
        psireal[realIdx] = X1;
        psiimag[realIdx] = Y1;
    }
}

//CZ

__global__ void CZ(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block1, QSIZE Block2)    //2^(qnum)           q9q8q7...q0 
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
    QSIZE corIdx, realIdx;
    if (Block1 >  Block2)
    {
        corIdx = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + idx % Block2;
    }
    else
    {
        corIdx = (idx / (Block2 / 2)) * 2 * Block2 + (idx % (Block2 / 2) / Block1) * 2 * Block1 + Block1 + idx % Block1;
    }
    realIdx = corIdx + Block2;
    if (realIdx < Dim)
    {
        psireal[realIdx] = -psireal[realIdx];
        psiimag[realIdx] = -psiimag[realIdx];
    }
}


//CR

__global__ void  CR(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block1,
    QSIZE Block2,
    STATE_T costheta,
    STATE_T sintheta)    //2^(qnum)           q9q8q7...q0 
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
    QSIZE realIdx;
    if (Block1 >  Block2)
    {
        realIdx = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + Block2 + idx%  Block2;
    }
    else
    {
        realIdx = (idx / (Block2 / 2)) * 2 * Block2 + Block2 + (idx % (Block2 / 2) / Block1) * 2 * Block1 + Block1 + idx%  Block1;
    }
    if (realIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T Y1 = psiimag[realIdx];
        psireal[realIdx] = X1*costheta - Y1*sintheta;
        psiimag[realIdx] = X1*sintheta + Y1*costheta;
    }
}

__global__ void controlCR(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block,
    QSIZE Block1,
    QSIZE Block2,
    QSIZE m,
    STATE_T costheta,
    STATE_T sintheta)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block[i];
            k = index % block[i];
            index = j * 2 * block[i] + block[i] + k;
        }
        QSIZE corIdx = index;
        STATE_T X1 = psireal[corIdx];                                  //11
        STATE_T Y1 = psiimag[corIdx];

        psireal[corIdx] = costheta*X1 - sintheta*Y1;
        psiimag[corIdx] = costheta*Y1 + sintheta*X1;
    }
}

__global__ void  iSWAP(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE Block1,
    QSIZE Block2, 
    STATE_T costheta,
    STATE_T sintheta)    //2^(qnum)           q9q8q7...q0 
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
    QSIZE corIdx, realIdx, temp;
    if (Block1 <  Block2)
    {
        temp = Block1;
        Block1 = Block2;
        Block2 = temp;
    }
    corIdx = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + idx%  Block2;
    realIdx = corIdx - Block1 + Block2;
    if (realIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];                                   //01
        STATE_T Y1 = psiimag[realIdx];
        STATE_T X2 = psireal[corIdx];                                  //10
        STATE_T Y2 = psiimag[corIdx];
        psireal[corIdx] = Y2;
        psiimag[corIdx] = -X2;
        psireal[realIdx] = costheta*X2-sintheta*Y1;
        psiimag[realIdx] = costheta*Y2+X1*sintheta;
    }
}

__global__ void controliSWAP(
    STATE_T * psireal,
    STATE_T * psiimag,
    QSIZE Dim,
    QSIZE * block,
    QSIZE Block1,
    QSIZE Block2,
    QSIZE m,
    STATE_T costheta,
    STATE_T sintheta)
{
    //to be update
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block[i];
            k = index % block[i];
            index = j * 2 * block[i] + block[i] + k;
        }
        QSIZE corIdx = index - Block2;
        QSIZE realIdx = index - Block1;

        STATE_T X1 = psireal[realIdx];                                   //01
        STATE_T Y1 = psiimag[realIdx];
        STATE_T X2 = psireal[corIdx];                                  //10
        STATE_T Y2 = psiimag[corIdx];
        psireal[corIdx] = Y2;
        psiimag[corIdx] = -X2;
        psireal[realIdx] = costheta*X2 - sintheta*Y1;
        psiimag[realIdx] = costheta*Y2 + X1*sintheta;
    }
}



__global__ void  qubitprob(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block, STATE_T *pr)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE bid = blockIdx.x, tid = threadIdx.x;
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;
    extern __shared__ STATE_T  dprob[];
    dprob[tid] = 0;
    int i;
    /*
    for (i = bid * blockDim.x + tid; i < Dim / 2; i += gridDim.x * THREADDIM)
    {
        //        QSIZE idx = bid*(psigpu.qnum-1))/THREADDIM.x + tid;
        //        QSIZE corIdx = idx / Block * 2 * Block + idx%Block;
        dprob[tid] += psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
    }
    */
    //dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];

    //ͬ�� ��֤ÿ�� thread ���Ѿ��ѽ��д�� shared[tid] ����
    if (corIdx < Dim)
    {
        dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];   //��ʡ��?
        __syncthreads();//��״�ӷ�
        int offset = 1, mask = 1;
        while (offset < THREADDIM)
        {
            if ((tid & mask) == 0)
            {
                dprob[tid] += dprob[tid + offset];
            }
            offset += offset;
            mask = offset + mask;
            __syncthreads();
        }
        //����ʱ��,��¼���,ֻ�� thread 0���� threadIdx.x =
        //dprob[0]��ͼ��õ����ĸ���
        if (tid == 0)
        {
            pr[bid] = dprob[0];
        }
    }
}//checked
__global__ void probsumnew1(STATE_T * psireal, STATE_T * psiimag, STATE_T *probtemp, size_t num1, size_t m, size_t Dim, size_t * block)
{
    size_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    size_t bid = blockIdx.x, tid = threadIdx.x;
    size_t index1, index = 0, index2, k, s;
    double temp = 0;
    index1 = num1 + idx;                              //index1��ʾidx��Ӧ�Ĳ���Ȩ��̬���
    if (index1 < (1u << m))
    {
        for (size_t j = 0; j < m; j++)
        {
            index += block[j] * ((index1 >> j) % 2);
        }//index ��ʾidx��Ӧ��̬�����
        for (size_t i = 0; i < Dim / (1u << m); i++)
        {
            index2 = i;
            for (size_t j = 0; j < m; j++)
            {
                s = index2 / block[j];
                k = index2 % block[j];
                index2 = s * 2 * block[j] + k;
            }
            index2 += index;
            temp += psireal[index2] * psireal[index2] + psiimag[index2] * psiimag[index2];
        }
        probtemp[idx] = temp;
    }
}
__global__ void  probsum(STATE_T * pr, STATE_T * prob)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    if (0 == idx)
    {
        STATE_T dprob = 0;
        for (int i = 0; i < gridDim.x; i++)
        {
            dprob += pr[i];
        }
        *prob = dprob;
    }
}//checked and can be optimized

__global__ void  qubitcollapse0(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block, STATE_T coef)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;
    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T Y1 = psiimag[realIdx];
        psireal[realIdx] = X1*coef;
        psireal[corIdx] = 0;
        psiimag[realIdx] = Y1*coef;
        psiimag[corIdx] = 0;
    }
}//checked

__global__ void  qubitcollapse1(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block, STATE_T coef)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;
    if (corIdx < Dim)
    {
        STATE_T X2 = psireal[corIdx];
        STATE_T Y2 = psiimag[corIdx];
        psireal[realIdx] = 0;
        psireal[corIdx] = X2*coef;
        psiimag[realIdx] = 0;
        psiimag[corIdx] = Y2*coef;
    }
}//checked








/*
__global__ void GATEGPU:: qbReset(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, QSIZE Block)      //reset to |0>, this operator is NOT UNITARY
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE BlockNum = idx / Block;
    QSIZE BlockInt = idx% Block;
    QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
    QSIZE corIdx = realIdx + Block;

    if (corIdx < Dim)
    {
        STATE_T X1 = psireal[realIdx];
        STATE_T Y1 = psiimag[realIdx];
        psireal[realIdx] = X1;
        psireal[corIdx] = 0;
        psiimag[realIdx] = Y1;
        psiimag[corIdx] = 0;
    }
}
*/


/**************************************************************************************
psireal:
psiimag:
pro:      save probability
block:    qubit number
m:        target qubit number
dec:      target qubit state





****************************************************************************************/
__global__ void  multiprob(STATE_T * psireal, STATE_T * psiimag,QSIZE Dim, STATE_T * pro, QSIZE * block, QSIZE m, QSIZE dec)
{
    QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
    QSIZE bid = blockIdx.x, tid = threadIdx.x;
    //QSIZE BlockNum = idx / Block;
    //QSIZE BlockInt = idx% Block;    
    extern __shared__ STATE_T dprob[];
    dprob[tid] = 0;
    QSIZE i, j, k;
    if (idx < Dim / (1 << m))
    {
        QSIZE index = idx;
        for (i = 0; i < m; i++)
        {
            j = index / block[i];
            k = index % block[i];
            index = j * 2 * block[i] + k;
        }                                                              //index Ŀ�����ȫΪ0
        QSIZE realIdx = index + dec;                                   //��Ҫ�ӵ�̬�ĸ���
        dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
		__syncthreads();//��״�ӷ�
        int offset = 1, mask = 1;
        while (offset < THREADDIM)
        {
            if ((tid & mask) == 0)
            {
                dprob[tid] += dprob[tid + offset];
            }
            offset += offset;
            mask = offset + mask;
            __syncthreads();
        }
        //����ʱ��,��¼���,ֻ�� thread 0���� threadIdx.x =
        //dprob[0]��ͼ��õ����ĸ���
        if (tid == 0)
        {
            pro[bid] = dprob[0];                       //�ټ���pro�ĺ;͵õ����ĸ���
        }
    }
}
}

/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Definition of Encapsulation of GPU gates
************************************************************************/

#define SET_BLOCKDIM  BLOCKDIM = (1 << (psigpu.qnum - 1)) / THREADDIM;

int GATEGPU:: devicecount()
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}


bool getSynchronizeResult(cudaError_t cudaStatue, char * pcGate)
{

    if (cudaSuccess != cudaStatue)
    {
        cout << "err " << pcGate << " = " << cudaGetErrorString(cudaStatue) << endl;
        return false;
    }
    return true;
}

#define GET_SYN_RES(x)      cudaError_t cudaStatue = cudaDeviceSynchronize();\
                            return getSynchronizeResult(cudaStatue,(x));

bool GATEGPU::destroyState(QState& psi, QState& psigpu,size_t stQnum)
{

    if ((nullptr == psi.real) ||
        (nullptr == psi.imag) ||
        (nullptr == psigpu.real) ||
        (nullptr == psigpu.imag))
    {
        return false;
    }

    if (stQnum<30)
    {
        cudaError_t cuda_statuss = cudaFree(psigpu.real);
        if (cudaSuccess != cuda_statuss)
        {
            cout << "psigpu.real free error" << endl;
            return false;
        }
        cuda_statuss =cudaFree(psigpu.imag);
        if (cudaSuccess != cuda_statuss)
        {
            cout << "psigpu.imag free error" << endl;
            return false;
        }
        free(psi.real);
        free(psi.imag);
    }
    else
    {
        cudaFreeHost(psigpu.real);
        cudaFreeHost(psigpu.imag);
    }


    return true;
}

bool GATEGPU::clearState(QState& psi, QState& psigpu,size_t stQnum)
{

    if ((nullptr == psi.real) ||
        (nullptr == psi.imag) ||
        (nullptr == psigpu.real) ||
        (nullptr == psigpu.imag))
    {
        return false;
    }

    if (stQnum<30)
    {
        QSIZE qsDim = (1ll << stQnum);
        memset(psi.real,0, qsDim *sizeof(STATE_T));
        memset(psi.imag, 0, qsDim * sizeof(STATE_T));
        psi.real[0] = 1;

        //cudaFree(psigpu.real);
        //cudaFree(psigpu.imag);
        cudaError_t cudaStatue = cudaMemcpy(psigpu.real, psi.real, sizeof(STATE_T)*qsDim, cudaMemcpyHostToDevice);
        if (cudaSuccess != cudaStatue)
        {
            cout << "psigpu real cudaMemcpy error" << endl;
        }
        cudaStatue = cudaMemcpy(psigpu.imag, psi.imag, sizeof(STATE_T)*qsDim, cudaMemcpyHostToDevice);
        if (cudaSuccess != cudaStatue)
        {
            cout << "psigpu imag cudaMemcpy error" << endl;
        }
    }
    else
    {
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
        gpu::initState << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum));
    }

    return true;
}

bool GATEGPU::initstate(QState& psi, QState& psigpu, int qnum)
{
    //QState psigpu;
    if (qnum >= 30)
    {
        cudaError_t cuda_status = cudaHostAlloc(&psi.real, sizeof(double)*(1ll << qnum), cudaHostAllocMapped);
        if (cuda_status != cudaSuccess)
        {
            printf("host alloc fail!\n");
            return false;
        }
        cudaError_t cuda_status1 = cudaHostAlloc(&psi.imag, sizeof(double)*(1ll << qnum), cudaHostAllocMapped);
        if (cuda_status1 != cudaSuccess)
        {
            printf("host alloc fail!\n");
            cudaFreeHost(psi.real);
            return false;
        }
        cudaHostGetDevicePointer(&psigpu.real, psi.real, 0);
        cudaHostGetDevicePointer(&psigpu.imag, psi.imag, 0);
        psi.qnum = qnum;
        psigpu.qnum = qnum;
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
		gpu::initState << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum));
        //std::cout << "err = " << cudaGetErrorString(cudaDeviceSynchronize()) << endl;
    }
    else 
    {
        QSIZE Dim = 1 << qnum;
        cudaError_t cuda_status;
        psi.real = (STATE_T*)malloc(Dim * sizeof(STATE_T));
        if (nullptr == psi.real)
        {
            printf("psi.real alloc memory error\n");
            return false;
        }
        psi.imag = (STATE_T*)malloc(Dim * sizeof(STATE_T));
        if (nullptr == psi.real)
        {
            printf("psi.imag alloc memory error\n");
            free(psi.imag);
            return false;
        }
        cuda_status = cudaMalloc((void**)&psigpu.real, sizeof(STATE_T)*Dim);
        if (cudaSuccess != cuda_status)
        {
            printf("psigpu.real alloc gpu memoery error!\n");
            free(psi.real);
            free(psi.imag);
            return false;
        }
        cuda_status = cudaMalloc((void**)&psigpu.imag, sizeof(STATE_T)*Dim);
        if (cudaSuccess != cuda_status)
        {
            printf("psigpu.imag alloc gpu memoery error!\n");
            free(psi.real);
            free(psi.imag);
            cudaFree(psigpu.real);
            return false;
        }
        
        memset(psi.real,0,Dim * sizeof(STATE_T));
        memset(psi.imag, 0, Dim * sizeof(STATE_T));
        psi.real[0] = 1;
        
        cuda_status = cudaMemcpy(psigpu.real, psi.real, sizeof(STATE_T)*Dim, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            printf("psigpu.imag alloc gpu cudaMemcpy error!\n");
            free(psi.real);
            free(psi.imag);
            cudaFree(psigpu.real);
            cudaFree(psigpu.imag);
            return false;
        }

        cuda_status = cudaMemcpy(psigpu.imag, psi.imag, sizeof(STATE_T)*Dim, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            printf("psigpu.imag alloc gpu cudaMemcpy error!\n");
            free(psi.real);
            free(psi.imag);
            cudaFree(psigpu.real);
            cudaFree(psigpu.imag);
            return false;
        }
        psigpu.qnum = qnum;
        psi.qnum = qnum;
    }

    return true;
}



bool GATEGPU:: Hadamard(QState& psigpu, size_t qn,bool isConjugate, double error_rate)

{
    if (gpu::randGenerator() > error_rate)
    {
        //QState* QPsi = (QState*)psi;
        //int BLOCKDIM = (1 << (psigpu.qnum - 1)) / THREADDIM;
        QSIZE BLOCKDIM = (1 << (psigpu.qnum - 1)) / THREADDIM;
		gpu::Hadamard<< <(BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn);
    }
    //GET_SYN_RES("Hadamard");
    cudaError_t cudaStatue = cudaDeviceSynchronize(); 
    return getSynchronizeResult(cudaStatue, "Hadamard");
}

//Hadamard gate
bool GATEGPU::Hadamardnew(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
    if (gpu::randGenerator() > error_rate)
    {
        
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
		gpu::Hadamardnew << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn);
    }

    GET_SYN_RES("Hadamardnew")
}

bool GATEGPU::controlHadamard(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlHadamard << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlHadamard")
}

//X gate
bool GATEGPU::X(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::X << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn);
	}

	GET_SYN_RES("X")
}

bool GATEGPU::controlX(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlX << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlX")
}


//Y gate
bool GATEGPU::Y(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::Y << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn);
	}

	GET_SYN_RES("Y")
}

bool GATEGPU::controlY(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlY << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlY")
}
//Z gate
bool GATEGPU::Z(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::Z << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn);
	}

	GET_SYN_RES("Z")
}

bool GATEGPU::controlZ(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlZ << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlZ")
}

//S gate
bool GATEGPU::S(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
	if (gpu::randGenerator() > error_rate)
	{
        int ilabel = 1;
        if (isConjugate)
        {
            ilabel = -1;
        }
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::S << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, ilabel);
	}

	GET_SYN_RES("S")
}

bool GATEGPU::controlS(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
        int ilabel = 1;
        if (isConjugate)
        {
            ilabel = -1;
        }
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlS << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m, ilabel);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlS")
}

//T gate
bool GATEGPU::T(QState& psigpu, size_t qn, bool isConjugate, double error_rate)

{
	if (gpu::randGenerator() > error_rate)
	{
        int ilabel = 1;
        if (isConjugate)
        {
            ilabel = -1;
        }
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::T << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, ilabel);
	}

	GET_SYN_RES("T")
}

bool GATEGPU::controlT(QState& psigpu, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
        int ilabel = 1;
        if (isConjugate)
        {
            ilabel = -1;
        }
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlT << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m, ilabel);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlT")
}
//RX
bool GATEGPU::RX(QState& psigpu, size_t qn, double theta, bool isConjugate,double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta,sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::RX << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, costheta, sintheta);
	}

	GET_SYN_RES("RX")
}

bool GATEGPU::controlRX(QState& psigpu, Qnum& qnum, double theta, bool isConjugate,double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlRX << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m, costheta, sintheta);
		cudaFreeHost(block);

	}
	GET_SYN_RES("controlRX")
}


//RY
bool GATEGPU::RY(QState& psigpu, size_t qn, double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::RY << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, costheta, sintheta);
	}

	GET_SYN_RES("RY")
}

bool GATEGPU::controlRY(QState& psigpu, Qnum& qnum, double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlRY << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m, costheta, sintheta);
		cudaFreeHost(block);

	}
	GET_SYN_RES("controlRY")
}

//RZ
bool GATEGPU::RZ(QState& psigpu, size_t qn, double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::RZ << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, costheta, sintheta);
	}

	GET_SYN_RES("RZ")
}

bool GATEGPU::controlRZ(QState& psigpu, Qnum& qnum, double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlRZ << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m, costheta, sintheta);
		cudaFreeHost(block);

	}
	GET_SYN_RES("controlRZ")
}

//CNOT
bool GATEGPU::CNOT(QState& psigpu, size_t qn0, size_t qn1, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::CNOT << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn0, 1 << qn1);
	
	}
	GET_SYN_RES("CNOT")
}

bool GATEGPU::controlCNOT(QState& psigpu, size_t qn0, size_t qn1, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlX << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlCNOT")
}

//CZ
bool GATEGPU::CZ(QState& psigpu, size_t qn0, size_t qn1, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::CZ << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn0, 1 << qn1);

	}
	GET_SYN_RES("CZ")
}

bool GATEGPU::controlCZ(QState& psigpu, size_t qn0, size_t qn1, Qnum& qnum, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		QSIZE m = qnum.size();
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlZ << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target, m);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlCZ")
}

//CR
bool GATEGPU::CR(QState& psigpu, size_t qn0, size_t qn1, double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
		double costheta, sintheta;
		if (isConjugate)
		{
			costheta = cos(-theta / 2);
			sintheta = sin(-theta / 2);
		}
		else
		{
			costheta = cos(theta / 2);
			sintheta = sin(theta / 2);
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::CR << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn0, 1 << qn1, costheta, sintheta);
	}
	GET_SYN_RES("CR")
}
//to be update
bool GATEGPU::controlCR(QState& psigpu, size_t qn0, size_t qn1, Qnum& qnum,double theta, bool isConjugate, double error_rate)
{
	if (gpu::randGenerator() > error_rate)
	{
        double costheta, sintheta;
        if (isConjugate)
        {
            costheta = cos(-theta);
            sintheta = sin(-theta);
        }
        else
        {
            costheta = cos(theta );
            sintheta = sin(theta );
        }
		QSIZE m = qnum.size();
        QSIZE control = qnum[m-2];
		QSIZE target = qnum.back();
		sort(qnum.begin(), qnum.end());
		QSIZE *block, *blockgpu;
		cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&blockgpu, block, 0);
		for (QSIZE i = 0; i < m; i++)
		{
			block[i] = 1 << qnum[i];
		}
		QSIZE BLOCKDIM;
		SET_BLOCKDIM
		gpu::controlCR << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
			(psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu,1<<control, 1 << target, m,costheta,sintheta);
		cudaFreeHost(block);
	}
	GET_SYN_RES("controlCR")
}

//iSWAP
bool GATEGPU::iSWAP(QState& psigpu, size_t qn0, size_t qn1, double theta, bool isConjugate, double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        double costheta, sintheta;
        if (isConjugate)
        {
            costheta = cos(-theta);
            sintheta = sin(-theta);
        }
        else
        {
            costheta = cos(theta);
            sintheta = sin(theta);
        }
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
		gpu::iSWAP << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn0, 1 << qn1, costheta, sintheta);
    }
    GET_SYN_RES("iSWAP")
}
//to be update
bool GATEGPU::controliSWAP(QState& psigpu, size_t qn0, size_t qn1, Qnum& qnum,double theta, bool isConjugate, double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        double costheta, sintheta;
        if (isConjugate)
        {
            costheta = cos(-theta);
            sintheta = sin(-theta);
        }
        else
        {
            costheta = cos(theta);
            sintheta = sin(theta);
        }
        QSIZE m = qnum.size();
        QSIZE target0 = qnum[m-2];
        QSIZE target1 = qnum.back();
        sort(qnum.begin(), qnum.end());
        QSIZE *block, *blockgpu;
        cudaHostAlloc((void **)&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&blockgpu, block, 0);
        for (QSIZE i = 0; i < m; i++)
        {
            block[i] = 1 << qnum[i];
        }
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
		gpu::controliSWAP << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), blockgpu, 1 << target0,1<<target1, m, costheta,sintheta);
        cudaFreeHost(block);
    }
    GET_SYN_RES("controliSWAP")
}


//unitary single gate
bool GATEGPU::unitarysingle(
    QState& psigpu,
    size_t qn,
    QState& matrix ,
    bool isConjugate,
    double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        
        if (isConjugate)
        {
			STATE_T temp_real, temp_imag;
            temp_real = matrix.real[1];
			temp_imag = matrix.imag[1];
            matrix.real[1] = matrix.real[2];
			matrix.imag[1] = matrix.imag[2];
            matrix.real[2] = temp_real;  //convert
			matrix.imag[2] = temp_imag;  //convert
            for (size_t i = 0; i < 4; i++)
            {
				matrix.real[i] = matrix.real[i];
				matrix.imag[i] = -matrix.imag[i];
               // matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        size_t kSingleGateElementNumber = 4;
        double *d_real = nullptr;
        double *d_imag = nullptr;

        cudaError_t cuda_status;
        cuda_status = cudaMalloc((void **)&d_real, sizeof(double) * kSingleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            return false;
        }

        cuda_status = cudaMalloc((void **)&d_imag, sizeof(double) * kSingleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            cudaFree(d_real);
            return false;
        }

        cuda_status = cudaMemcpy(d_real, matrix.real, sizeof(double) * kSingleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        cuda_status = cudaMemcpy(d_imag, matrix.imag, sizeof(double) * kSingleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        QSIZE BLOCKDIM;
		SET_BLOCKDIM

		gpu::unitarysingle << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, d_real, d_imag);
        cuda_status = cudaDeviceSynchronize();

        cudaFree(d_real);
        cudaFree(d_imag);
        return getSynchronizeResult(cuda_status, "unitarysingle");
    }

    return true;
}

bool GATEGPU::controlunitarysingle(
    QState& psigpu,
    Qnum& qnum,
    QState& matrix,
    bool isConjugate,
    double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        double costheta, sintheta;
        if (isConjugate)
        {
			STATE_T temp_real, temp_imag;
			temp_real = matrix.real[1];
			temp_imag = matrix.imag[1];
			matrix.real[1] = matrix.real[2];
			matrix.imag[1] = matrix.imag[2];
			matrix.real[2] = temp_real;  //convert
			matrix.imag[2] = temp_imag;  //convert
            for (size_t i = 0; i < 4; i++)
            {
				matrix.real[i] = matrix.real[i];
				matrix.imag[i] = -matrix.imag[i];
                //matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        cudaError_t cuda_status;
        QSIZE m = qnum.size();
        QSIZE target = qnum.back();
        sort(qnum.begin(), qnum.end());

        QSIZE *d_block = nullptr;
        QSIZE *h_block = nullptr;
        h_block = (QSIZE *)malloc(sizeof(QSIZE)*m);
        if (nullptr == h_block)
        {
            fprintf(stderr, "malloc error\n");
            return false;
        }

        for (QSIZE i = 0; i < m; i++)
        {
            h_block[i] = 1 << qnum[i];
        }

        cuda_status = cudaMalloc((void **)&d_block, sizeof(QSIZE)*m);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            return false;
        }

        cuda_status = cudaMemcpy(d_block, h_block, sizeof(QSIZE) * m, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            return false;
        }

        size_t kSingleGateElementNumber = 4;
        double *d_real = nullptr;
        double *d_imag = nullptr;

        cuda_status = cudaMalloc((void **)&d_real, sizeof(double) * kSingleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            cudaFree(d_block);
            return false;
        }

        cuda_status = cudaMalloc((void **)&d_imag, sizeof(double) * kSingleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            return false;
        }

        cuda_status = cudaMemcpy(d_real, matrix.real, sizeof(double) * kSingleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        cuda_status = cudaMemcpy(d_imag, matrix.imag, sizeof(double) * kSingleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        QSIZE BLOCKDIM;
        SET_BLOCKDIM

        BLOCKDIM = (1 << (psigpu.qnum - 1)) / THREADDIM;
		gpu::controlunitarysingle << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), d_block, 1 << target, m,d_real, d_imag);
        cuda_status = cudaDeviceSynchronize();

        free(h_block);
        cudaFree(d_block);
        cudaFree(d_real);
        cudaFree(d_imag);

        return getSynchronizeResult(cuda_status, "controlunitarysingle");
    }
    return true;
}


//unitary double gate
bool GATEGPU::unitarydouble(
    QState& psigpu,
    size_t qn_0, 
    size_t qn_1,
    QState& matrix,
    bool isConjugate,
    double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {

        if (isConjugate)
        {
			STATE_T temp_real,temp_imag;
            for (size_t i = 0; i < 4; i++)
            {
                for (size_t j = i + 1; j < 4; j++)
                {
					temp_real = matrix.real[4 * i + j];
					temp_imag = matrix.imag[4 * i + j];
                    matrix.real[4 * i + j] = matrix.real[4 * j + i];
					matrix.imag[4 * i + j] = matrix.imag[4 * j + i];
                    matrix.real[4 * j + i] = temp_real;
					matrix.imag[4 * j + i] = temp_imag;
                }
            }
            for (size_t i = 0; i < 16; i++)
            {
                //matrix[i].imag = -matrix[i].imag;
				matrix.real[i] = matrix.real[i];
				matrix.imag[i] = -matrix.imag[i];
;                //matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        size_t kDoubleGateElementNumber = 16;
        double *d_real = nullptr;
        double *d_imag = nullptr;
        cudaError_t cuda_status;

        cuda_status = cudaMalloc((void **)&d_real, sizeof(double) * kDoubleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            return false;
        }

        cuda_status = cudaMalloc((void **)&d_imag, sizeof(double) * kDoubleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            cudaFree(d_real);
            return false;
        }

        cuda_status = cudaMemcpy(d_real, matrix.real, sizeof(double) * kDoubleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        cuda_status = cudaMemcpy(d_imag, matrix.imag, sizeof(double) * kDoubleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
        gpu::unitarydouble << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn_0, 1 << qn_1, d_real, d_imag);
        cuda_status = cudaDeviceSynchronize();

        cudaFree(d_real);
        cudaFree(d_imag);
        return getSynchronizeResult(cuda_status, "unitarydouble");
    }

    return true;
}

bool GATEGPU::controlunitarydouble(
    QState& psigpu,
    Qnum& qnum,
    QState& matrix,
    bool isConjugate,
    double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        double costheta, sintheta;
        if (isConjugate)
        {
			STATE_T temp_real, temp_imag;
            for (size_t i = 0; i < 4; i++)
            {
                for (size_t j = i + 1; j < 4; j++)
                {
					temp_real = matrix.real[4 * i + j];
					temp_imag = matrix.imag[4 * i + j];
					matrix.real[4 * i + j] = matrix.real[4 * j + i];
					matrix.imag[4 * i + j] = matrix.imag[4 * j + i];
					matrix.real[4 * j + i] = temp_real;
					matrix.imag[4 * j + i] = temp_imag;
                }
            }
            for (size_t i = 0; i < 16; i++)
            {
				matrix.real[i] = matrix.real[i];
				matrix.imag[i] = -matrix.imag[i];
                //matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        QSIZE m = qnum.size();
        QSIZE target0 = qnum[m-2];
        QSIZE target1 = qnum.back();
        sort(qnum.begin(), qnum.end());
        QSIZE *h_block, *d_block;

        h_block = (QSIZE *)malloc(sizeof(QSIZE) * m);
        if (nullptr == h_block)
        {
            fprintf(stderr, "malloc error\n");
            return false;
        }

        for (QSIZE i = 0; i < m; i++)
        {
            h_block[i] = 1 << qnum[i];
        }

        cudaError_t cuda_status;
        cuda_status = cudaMalloc((void **)&d_block, sizeof(QSIZE) * m);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            return false;
        }

        cuda_status = cudaMemcpy(d_block, h_block, sizeof (QSIZE) * m, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            return false;
        }

        size_t kDoubleGateElementNumber = 16;
        double *d_real = nullptr;
        double *d_imag = nullptr;

        cuda_status = cudaMalloc((void **)&d_real, sizeof(double) * kDoubleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            cudaFree(d_block);
            return false;
        }

        cuda_status = cudaMalloc((void **)&d_imag, sizeof(double) * kDoubleGateElementNumber);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            return false;
        }

        cuda_status = cudaMemcpy(d_real, matrix.real, sizeof(double) * kDoubleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        cuda_status = cudaMemcpy(d_imag, matrix.imag, sizeof(double) * kDoubleGateElementNumber, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            free(h_block);
            cudaFree(d_block);
            cudaFree(d_real);
            cudaFree(d_imag);
            return false;
        }

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
        gpu::controlunitarydouble << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > 
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), d_block,
                1 << target0,1<<target1, m, d_real, d_imag);
        cuda_status = cudaDeviceSynchronize();

        free(h_block);
        cudaFree(d_block);
        cudaFree(d_real);
        cudaFree(d_imag);
        return getSynchronizeResult(cuda_status, "controlunitarydouble");
    }
    
    return true;
}



//qbReset
bool GATEGPU::qbReset(QState& psigpu, size_t qn, double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        cudaError_t cuda_status;
        double * resultgpu;
        // cudaHostAlloc((void **)&result, sizeof(STATE_T)*(psigpu.qnum-1))/THREADDIM, cudaHostAllocMapped);
        //cudaHostGetDevicePointer(&resultgpu, result, 0);

        cuda_status = cudaMalloc((void **)&resultgpu, sizeof(STATE_T)*(1 << (psigpu.qnum - 1)) / THREADDIM);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            return false;
        }

        double * probgpu, *prob;
        cuda_status = cudaHostAlloc((void **)&prob, sizeof(STATE_T), cudaHostAllocMapped);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            cudaFree(resultgpu);
        }

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
		gpu::qubitprob << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM, THREADDIM * sizeof(STATE_T) >> >
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, resultgpu);    //���ʵ�һ�ι�Լ
        cuda_status = cudaDeviceSynchronize();
		gpu::probsum << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (resultgpu, probgpu);                   //Ҫ������̬�ĸ��ʴ���prob��
        cuda_status = cudaDeviceSynchronize();           //�ȸ�����ȫ�������
        *prob = 1 / sqrt(*prob);
        gpu::qubitcollapse0 << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > 
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, *prob);
        cuda_status = cudaDeviceSynchronize();

        cudaFree(resultgpu);
        cudaFreeHost(prob);
        return getSynchronizeResult(cuda_status, "qReset");
    }

    return true;
}

int GATEGPU::qubitmeasure(QState& psigpu, QSIZE Block)
{

    // double * result;
    double * resultgpu;
    // cudaHostAlloc((void **)&result, sizeof(STATE_T)*(psigpu.qnum-1))/THREADDIM, cudaHostAllocMapped);
    //cudaHostGetDevicePointer(&resultgpu, result, 0);
    //QSIZE BLOCKDIM = (1 << (psigpu.qnum - 1)) / THREADDIM;
    QSIZE BLOCKDIM;
    SET_BLOCKDIM

    int count = (0 == BLOCKDIM) ? 1 : BLOCKDIM;
    cudaError_t cuda_status = cudaMalloc(&resultgpu, sizeof(STATE_T)* count);
    if (cudaSuccess != cuda_status)
    {
        cout << "resultgpu  " << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }
    double * probgpu, prob;
    //cudaHostAlloc((void **)&prob, sizeof(STATE_T), cudaHostAllocMapped);
    //cudaHostGetDevicePointer(&probgpu, prob, 0);
    cuda_status = cudaMalloc(&probgpu, sizeof(STATE_T));
    if (cudaSuccess != cuda_status)
    {
        cout << "probgpu  " << cudaGetErrorString(cuda_status) << endl;
        cudaFree(resultgpu);
        return -1;
    }
    gpu::qubitprob << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM, THREADDIM * sizeof(STATE_T) >> > 
        (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), Block, resultgpu);    //���ʵ�һ�ι�Լ
    cuda_status = cudaDeviceSynchronize();           //�ȸ�����ȫ�������
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        cudaFree(resultgpu);
        cudaFree(probgpu);
        return -1;
    }
    //double *prob;
	gpu::probsum << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (resultgpu, probgpu);                   //Ҫ������̬�ĸ��ʴ���prob��
    cuda_status = cudaDeviceSynchronize();           //�ȸ�����ȫ�������
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        cudaFree(resultgpu);
        cudaFree(probgpu);
        return -1;
    }

    cuda_status = cudaMemcpy(&prob, probgpu, sizeof(STATE_T), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_status)
    {
        fprintf(stderr, "cudaMemcpy error\n");
        cudaFree(resultgpu);
        cudaFree(probgpu);
        return -1;
    }

    cuda_status = cudaDeviceSynchronize();           //�ȸ�����ȫ�������
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        cudaFree(resultgpu);
        cudaFree(probgpu);
        return -1;
    }

    int outcome = 0;
    if (gpu::randGenerator() > prob)
    {
        outcome = 1;
    }

    if (0 == outcome)
    {

        prob = 1 / sqrt(prob);
        gpu::qubitcollapse0 <<< (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >>>
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), Block, prob);
        //GET_SYN_RES("qubitmeasure")
    }
    else
    {

        prob = 1 / sqrt(1 - prob);
        gpu::qubitcollapse1 <<< (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >>>
            (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), Block, prob);
        //GET_SYN_RES("qubitmeasure")
    }
    cuda_status = cudaDeviceSynchronize();
    cudaFree(probgpu);
    cudaFree(resultgpu);
    getSynchronizeResult(cuda_status, "qubitmeasure");

    return outcome;
}//checked


bool probcompare(pair<size_t, double>& a, pair<size_t, double>& b)
{
    return a.second> b.second;
}

bool GATEGPU::pMeasurenew(QState& psigpu, vector<pair<size_t, double>>& vprob, Qnum& qnum)
{
    cudaDeviceSynchronize();
    QSIZE m = qnum.size();
    sort(qnum.begin(), qnum.end());
    cudaError_t cuda_status;

    if (m <= psigpu.qnum / 2)
    {
        QSIZE *block, *blockgpu;
        cuda_status = cudaHostAlloc(&block, sizeof(QSIZE)*m, cudaHostAllocMapped);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            return false;
        }

        cudaHostGetDevicePointer(&blockgpu, block, 0);
        QSIZE temp;
        for (size_t i = 0; i < m; i++)
        {
            block[i] = 1 << qnum[i];
        }//����
        double *probgpu;
        double * probc, *result;
        cuda_status = cudaHostAlloc(&probc, sizeof(STATE_T)*(1 << m), cudaHostAllocMapped);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaHostAlloc fail!\n");
            cudaFreeHost(block);
            return false;
        }
        cudaHostGetDevicePointer(&probgpu, probc, 0);
        //probc=(STATE_T*)malloc(sizeof(STATE_T)*(1<<m));          //����̬�ĸ���
        QSIZE *blockgpu1;                       //block
        QSIZE BLOCKDIM = (1u << (psigpu.qnum - 1)) / THREADDIM;
        cuda_status = cudaMalloc((&blockgpu1), sizeof(QSIZE)*m);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMallco error\n");
            cudaFreeHost(block);
            cudaFreeHost(probc);
            return false;
        }

        cuda_status = cudaMalloc((&result), sizeof(double)*(BLOCKDIM == 0 ? 1 : BLOCKDIM));          //����ʵ��м����
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMallco error\n");
            cudaFreeHost(block);
            cudaFreeHost(probc);
            cudaFree(blockgpu1);
            return false;
        }

        cuda_status = cudaMemcpy(blockgpu1, block, sizeof(QSIZE)*m, cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMallco error\n");
            cudaFreeHost(block);
            cudaFreeHost(probc);
            cudaFree(result);
            cudaFree(blockgpu1);
            return false;
        }

        for (size_t i = 0; i < 1u << m; i++)
        {
            size_t index = 0;
            for (size_t j = 0; j < m; j++)
            {
                index += block[j] * ((i >> j) % 2);
            }
            // cout << "index\t" << index << endl;
			gpu::multiprob << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM, THREADDIM * sizeof(STATE_T) >> >
                (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), result, blockgpu1, m, index);
			gpu::probsum << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (result, probgpu + i);
            //cudaMemcpy((void GATEGPU::*)probc, (void GATEGPU::*)probgpu, sizeof(STATE_T)*(1<<m), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            //vprob[i].first = i;
            //vprob[i].second = probc[i];
            vprob.push_back(GPUPAIR(i, probc[i]));

        }
        if (vprob.size() <= 10)
        {
            sort(vprob.begin(), vprob.end(), probcompare);
        }
        else
        {
            sort(vprob.begin(), vprob.end(), probcompare);
            vprob.erase(vprob.begin() + 10, vprob.end());
        }
        //std::cout << *probc << endl;
        cudaFree(result);
        cudaFree(blockgpu1);
        cudaFreeHost(probc);
        cudaFreeHost(block);
        return true;
    }
    else
    {
        size_t Dim = 1u << psigpu.qnum;
        size_t blocknum = 1u << (m - psigpu.qnum / 4);         //blocknum��ʾblock��
        STATE_T *probtemp, *probtempgpu;
        cuda_status = cudaHostAlloc(&probtemp, sizeof(double) * blocknum, cudaHostAllocMapped);

        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            return false;
        }

        cuda_status = cudaHostGetDevicePointer(&probtempgpu, probtemp, 0);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            cudaFreeHost(probtemp);
            return false;
        }

        size_t *block, *blockgpu;
        cuda_status = cudaHostAlloc(&block, sizeof(size_t)*m, cudaHostAllocMapped);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            cudaFreeHost(probtemp);
            return false;
        }

        cuda_status = cudaHostGetDevicePointer(&blockgpu, block, 0);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            cudaFreeHost(probtemp);
            cudaFreeHost(block);
            return false;
        }
        for (size_t i = 0; i < m; i++)
        {
            block[i] = 1u << qnum[i];
        }//����
        for (size_t i = 0; i < blocknum; i++)
        {
            probtemp[i] = 0;
        }
        for (size_t i = 0; i < 10; i++)
        {
            vprob.push_back(GPUPAIR(0, 0));
        }
        size_t BLOCKDIM = blocknum / THREADDIM;
        for (size_t i = 0; i < (1u << m); i += blocknum)
        {
            //(STATE_T * psireal, STATE_T * psiimag, STATE_T *probtemp, size_t num1, size_t m, size_t Dim, size_t * block)

            gpu::probsumnew1 << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > 
                (psigpu.real, psigpu.imag, probtempgpu, i, m, Dim, blockgpu);

            cuda_status = cudaDeviceSynchronize();
            if (cudaSuccess != cuda_status)
            {
                cout << "error" << endl;
            }
            for (size_t j = 0; j < blocknum; j++)
            {
                if (probtemp[j] > vprob[9].second)
                {
                    vprob[9] = GPUPAIR(i + j, probtemp[j]);
                    sort(vprob.begin(), vprob.end(), probcompare);
                }
            }
        }
        cudaFreeHost(probtemp);
        cudaFreeHost(block);
        return true;
    }

}


bool GATEGPU::getState(QState &psi, QState &psigpu, int qnum)
{
    cudaError_t cuda_status;

    if (qnum < 30)
    {
        QSIZE Dim = 1 << qnum;
        cuda_status = cudaMemcpy(psi.real, psigpu.real, sizeof(STATE_T)*Dim, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            return false;
        }

        cuda_status = cudaMemcpy(psi.imag, psigpu.imag, sizeof(STATE_T)*Dim, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            return false;
        }
    }
    return true;
}

double gpu::randGenerator()
{
    int  ia = 16807, im = 2147483647, iq = 127773, ir = 2836;           /*difine constant number in 16807 generator.*/
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    static int irandseed = timeinfo->tm_year + 70 *
        (timeinfo->tm_mon + 1 + 12 *
        (timeinfo->tm_mday + 31 *
            (timeinfo->tm_hour + 23 *
            (timeinfo->tm_min + 59 * timeinfo->tm_sec))));
    static int irandnewseed;
    if (ia*(irandseed%iq) - ir * (irandseed / iq) >= 0)
        irandnewseed = ia * (irandseed%iq) - ir * (irandseed / iq);
    else
        irandnewseed = ia * (irandseed%iq) - ir * (irandseed / iq) + im;
    irandseed = irandnewseed;
    return (double)irandnewseed / im;

}













//bool GATEGPU:: toffoli(QState& psigpu, size_t qn0, size_t qn1, size_t qn2, double error_rate)
//{
//    if (randGenerator() > error_rate)
//    {
//        //QState* QPsi = (QState*)psi;
//        QSIZE BLOCKDIM;
//        SET_BLOCKDIM
//        toffoli << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag,1<<(psigpu.qnum), 1 << qn0, 1 << qn1, 1 << qn2);
//    }
//    GET_SYN_RES("toffoli")
//}
//bool GATEGPU:: toffolidagger(QState& psigpu, size_t qn0, size_t qn1, size_t qn2, double error_rate)
//{
//    if (randGenerator() > error_rate)
//    {
//        //QState* QPsi = (QState*)psi;
//        QSIZE BLOCKDIM;
//        SET_BLOCKDIM
//        toffolidagger << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag,1<<(psigpu.qnum), 1 << qn0, 1 << qn1, 1 << qn2);
//    }
//    GET_SYN_RES("toffolidagger")
//}
//bool GATEGPU:: qbReset(QState& psigpu, size_t qn, double error_rate)
//{
//    if (randGenerator() > error_rate)
//    {
//        double * resultgpu;
//        // cudaHostAlloc((void **)&result, sizeof(STATE_T)*(psigpu.qnum-1))/THREADDIM, cudaHostAllocMapped);
//        //cudaHostGetDevicePointer(&resultgpu, result, 0);
//        cudaMalloc((void **)&resultgpu, sizeof(STATE_T)*(1 << (psigpu.qnum - 1)) / THREADDIM);
//        double * probgpu, *prob;
//        cudaHostAlloc((void **)&prob, sizeof(STATE_T), cudaHostAllocMapped);
//        cudaHostGetDevicePointer(&probgpu, prob, 0);
//        QSIZE BLOCKDIM;
//        SET_BLOCKDIM
//        qubitprob << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM, THREADDIM * sizeof(STATE_T) >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, resultgpu);    //���ʵ�һ�ι�Լ
//        probsum << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (resultgpu, probgpu);                   //Ҫ������̬�ĸ��ʴ���prob��
//        cudaDeviceSynchronize();           //�ȸ�����ȫ�������
//        *prob = 1 / sqrt(*prob);
//        qubitcollapse0 << < (BLOCKDIM == 0 ? 1 : BLOCKDIM), THREADDIM >> > (psigpu.real, psigpu.imag, 1 << (psigpu.qnum), 1 << qn, *prob);
//        cudaDeviceSynchronize();           
//        cudaFree(resultgpu);
//        cudaFreeHost(prob);
//       // std::cout << "err = " << cudaGetErrorString(cudaDeviceSynchronize()) << endl;
//    }
//    GET_SYN_RES("qbReset")
//    
//}
//
//




 






