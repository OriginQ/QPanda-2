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

#include <vector>
#include <algorithm>
#include <time.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "GPUGates.h"
using namespace std;

namespace gpu {
    __global__ void
        unitarysingle(
            STATE_T * psireal,
            STATE_T * psiimag,
            QSIZE Dim,
            QSIZE Block,
            STATE_T matrix_real00,
            STATE_T matrix_real01,
            STATE_T matrix_real10,
            STATE_T matrix_real11,
            STATE_T matrix_imag00,
            STATE_T matrix_imag01,
            STATE_T matrix_imag10,
            STATE_T matrix_imag11)
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
            psireal[realIdx] = matrix_real00 * X1 - matrix_imag00 * Y1 + matrix_real01 * X2 - matrix_imag01 * Y2;
            psireal[corIdx] = matrix_real10 * X1 - matrix_imag10 * Y1 + matrix_real11 * X2 - matrix_imag11 * Y2;
            psiimag[realIdx] = matrix_real00 * Y1 + matrix_imag00 * X1 + matrix_real01 * Y2 + matrix_imag01 * X2;
            psiimag[corIdx] = matrix_real10 * Y1 + matrix_imag10 * X1 + matrix_real11 * Y2 + matrix_imag11 * X2;
        }
    }

    __global__ void controlunitarysingle(
        STATE_T * psireal,
        STATE_T * psiimag,
        QSIZE Dim,
        QSIZE target_qubit,
        QSIZE controller_mask,
        STATE_T matrix_real00,
        STATE_T matrix_real01,
        STATE_T matrix_real10,
        STATE_T matrix_real11,
        STATE_T matrix_imag00,
        STATE_T matrix_imag01,
        STATE_T matrix_imag10,
        STATE_T matrix_imag11
    )
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

        if (
            idx < Dim && 
            ((idx & controller_mask) == controller_mask) &&
            ((idx & target_qubit) == target_qubit)
           )
        {
            QSIZE corIdx = idx;                                     //1
            QSIZE realIdx = corIdx - target_qubit;                  //0
            STATE_T X1 = psireal[realIdx];
            STATE_T X2 = psireal[corIdx];
            STATE_T Y1 = psiimag[realIdx];
            STATE_T Y2 = psiimag[corIdx];
            psireal[realIdx] = matrix_real00 * X1 - matrix_imag00 * Y1 + matrix_real01 * X2 - matrix_imag01 * Y2;
            psireal[corIdx] = matrix_real10 * X1 - matrix_imag10 * Y1 + matrix_real11 * X2 - matrix_imag11 * Y2;
            psiimag[realIdx] = matrix_real00 * Y1 + matrix_imag00 * X1 + matrix_real01 * Y2 + matrix_imag01 * X2;
            psiimag[corIdx] = matrix_real10 * Y1 + matrix_imag10 * X1 + matrix_real11 * Y2 + matrix_imag11 * X2;
        }
    }

    __global__ void
        unitarydouble(
            STATE_T * psireal,
            STATE_T * psiimag,
            QSIZE Dim,
            QSIZE Block1,
            QSIZE Block2,
            STATE_T real0000,
            STATE_T real0001,
            STATE_T real0010,
            STATE_T real0011,
            STATE_T real0100,
            STATE_T real0101,
            STATE_T real0110,
            STATE_T real0111,
            STATE_T real1000,
            STATE_T real1001,
            STATE_T real1010,
            STATE_T real1011,
            STATE_T real1100,
            STATE_T real1101,
            STATE_T real1110,
            STATE_T real1111,
            STATE_T imag0000,
            STATE_T imag0001,
            STATE_T imag0010,
            STATE_T imag0011,
            STATE_T imag0100,
            STATE_T imag0101,
            STATE_T imag0110,
            STATE_T imag0111,
            STATE_T imag1000,
            STATE_T imag1001,
            STATE_T imag1010,
            STATE_T imag1011,
            STATE_T imag1100,
            STATE_T imag1101,
            STATE_T imag1110,
            STATE_T imag1111)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
        QSIZE Idx00, Idx01, Idx10, Idx11;
        if (Block1 > Block2)
        {
            Idx10 = (idx / (Block1 / 2)) * 2 * Block1 + Block1 + (idx % (Block1 / 2) / Block2) * 2 * Block2 + idx % Block2;
        }
        else
        {
            Idx10 = (idx / (Block2 / 2)) * 2 * Block2 + (idx % (Block2 / 2) / Block1) * 2 * Block1 + Block1 + idx % Block1;
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
            psireal[Idx00] = real0000 * X00 - imag0000 * Y00
                + real0001 * X01 - imag0001 * Y01
                + real0010 * X10 - imag0010 * Y10
                + real0011 * X11 - imag0011 * Y11;
            psiimag[Idx00] = imag0000 * X00 + real0000 * Y00
                + imag0001 * X01 + real0001 * Y01
                + imag0010 * X10 + real0010 * Y10
                + imag0011 * X11 + real0011 * Y11;

            psireal[Idx01] = real0100 * X00 - imag0100 * Y00
                + real0101 * X01 - imag0101 * Y01
                + real0110 * X10 - imag0110 * Y10
                + real0111 * X11 - imag0111 * Y11;
            psiimag[Idx01] = imag0100 * X00 + real0100 * Y00
                + imag0101 * X01 + real0101 * Y01
                + imag0110 * X10 + real0110 * Y10
                + imag0111 * X11 + real0111 * Y11;

            psireal[Idx10] = real1000 * X00 - imag1000 * Y00
                + real1001 * X01 - imag1001 * Y01
                + real1010 * X10 - imag1010 * Y10
                + real1011 * X11 - imag1011 * Y11;
            psiimag[Idx10] = imag1000 * X00 + real1000 * Y00
                + imag1001 * X01 + real1001 * Y01
                + imag1010 * X10 + real1010 * Y10
                + imag1011 * X11 + real1011 * Y11;
            psireal[Idx11] = real1100 * X00 - imag1100 * Y00
                + real1101 * X01 - imag1101 * Y01
                + real1110 * X10 - imag1110 * Y10
                + real1111 * X11 - imag1111 * Y11;
            psiimag[Idx11] = imag1100 * X00 + real1100 * Y00
                + imag1101 * X01 + real1101 * Y01
                + imag1110 * X10 + real1110 * Y10
                + imag1111 * X11 + real1111 * Y11;
        }
    }

    __global__ void controlunitarydouble(
        STATE_T * psireal,
        STATE_T * psiimag,
        QSIZE Dim,
        QSIZE controller_mask,
        QSIZE control_qubit,
        QSIZE target_qubit,
        STATE_T real0000,
        STATE_T real0001,
        STATE_T real0010,
        STATE_T real0011,
        STATE_T real0100,
        STATE_T real0101,
        STATE_T real0110,
        STATE_T real0111,
        STATE_T real1000,
        STATE_T real1001,
        STATE_T real1010,
        STATE_T real1011,
        STATE_T real1100,
        STATE_T real1101,
        STATE_T real1110,
        STATE_T real1111,
        STATE_T imag0000,
        STATE_T imag0001,
        STATE_T imag0010,
        STATE_T imag0011,
        STATE_T imag0100,
        STATE_T imag0101,
        STATE_T imag0110,
        STATE_T imag0111,
        STATE_T imag1000,
        STATE_T imag1001,
        STATE_T imag1010,
        STATE_T imag1011,
        STATE_T imag1100,
        STATE_T imag1101,
        STATE_T imag1110,
        STATE_T imag1111)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        if (
            idx < Dim &&
            ((idx & controller_mask) == controller_mask) &&
            ((idx & control_qubit) == control_qubit) &&
            ((idx & target_qubit) == target_qubit)
            )
        {
            QSIZE Idx00 = idx - control_qubit - target_qubit;
            QSIZE Idx01 = Idx00 - control_qubit;
            QSIZE Idx10 = Idx00 - target_qubit;
            QSIZE Idx11 = idx;
            STATE_T X00 = psireal[Idx00];
            STATE_T X01 = psireal[Idx01];
            STATE_T X10 = psireal[Idx10];
            STATE_T X11 = psireal[Idx11];
            STATE_T Y00 = psiimag[Idx00];
            STATE_T Y01 = psiimag[Idx01];
            STATE_T Y10 = psiimag[Idx10];
            STATE_T Y11 = psiimag[Idx11];
            psireal[Idx00] = real0000 * X00 - imag0000 * Y00
                + real0001 * X01 - imag0001 * Y01
                + real0010 * X10 - imag0010 * Y10
                + real0011 * X11 - imag0011 * Y11;
            psiimag[Idx00] = imag0000 * X00 + real0000 * Y00
                + imag0001 * X01 + real0001 * Y01
                + imag0010 * X10 + real0010 * Y10
                + imag0011 * X11 + real0011 * Y11;

            psireal[Idx01] = real0100 * X00 - imag0100 * Y00
                + real0101 * X01 - imag0101 * Y01
                + real0110 * X10 - imag0110 * Y10
                + real0111 * X11 - imag0111 * Y11;
            psiimag[Idx01] = imag0100 * X00 + real0100 * Y00
                + imag0101 * X01 + real0101 * Y01
                + imag0110 * X10 + real0110 * Y10
                + imag0111 * X11 + real0111 * Y11;

            psireal[Idx10] = real1000 * X00 - imag1000 * Y00
                + real1001 * X01 - imag1001 * Y01
                + real1010 * X10 - imag1010 * Y10
                + real1011 * X11 - imag1011 * Y11;
            psiimag[Idx10] = imag1000 * X00 + real1000 * Y00
                + imag1001 * X01 + real1001 * Y01
                + imag1010 * X10 + real1010 * Y10
                + imag1011 * X11 + real1011 * Y11;
            psireal[Idx11] = real1100 * X00 - imag1100 * Y00
                + real1101 * X01 - imag1101 * Y01
                + real1110 * X10 - imag1110 * Y10
                + real1111 * X11 - imag1111 * Y11;
            psiimag[Idx11] = imag1100 * X00 + real1100 * Y00
                + imag1101 * X01 + real1101 * Y01
                + imag1110 * X10 + real1110 * Y10
                + imag1111 * X11 + real1111 * Y11;
        }
    }

    __global__ void  initState(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

        if (idx < Dim && idx != 0)
        {
            psireal[idx] = 0;
            psiimag[idx] = 0;
        }
        if (0 == idx)
        {
            psireal[0] = 1;
            psiimag[0] = 0;
        }
    }

    __global__ void qubitprob(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, STATE_T *pr)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;
        QSIZE bid = blockIdx.x, tid = threadIdx.x;
        QSIZE BlockNum = idx / Block;
        QSIZE BlockInt = idx % Block;
        QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
        QSIZE corIdx = realIdx + Block;
        extern __shared__ STATE_T  dprob[];
        dprob[tid] = 0;

        if (corIdx < Dim)
        {
            dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
            __syncthreads();
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
            if (tid == 0)
            {
                pr[bid] = dprob[0];
            }
        }
    }

    __global__ void probsumnew1(STATE_T * psireal, STATE_T * psiimag, STATE_T *probtemp, size_t num1, size_t m, size_t Dim, size_t * block)
    {
        size_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        size_t index1, index = 0, index2, k, s;
        double temp = 0;
        index1 = num1 + idx;                              //index1表示idx对应的不加权的态序号
        if (index1 < (1u << m))
        {
            for (size_t j = 0; j < m; j++)
            {
                index += block[j] * ((index1 >> j) % 2);
            }//index 表示idx对应的态的序号
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

    __global__ void  qubitcollapse0(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, STATE_T coef)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        QSIZE BlockNum = idx / Block;
        QSIZE BlockInt = idx % Block;
        QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
        QSIZE corIdx = realIdx + Block;
        if (corIdx < Dim)
        {
            STATE_T X1 = psireal[realIdx];
            STATE_T Y1 = psiimag[realIdx];
            psireal[realIdx] = X1 * coef;
            psireal[corIdx] = 0;
            psiimag[realIdx] = Y1 * coef;
            psiimag[corIdx] = 0;
        }
    }//checked

    __global__ void  qubitcollapse1(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim, QSIZE Block, STATE_T coef)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        QSIZE BlockNum = idx / Block;
        QSIZE BlockInt = idx % Block;
        QSIZE realIdx = BlockNum * 2 * Block + BlockInt;
        QSIZE corIdx = realIdx + Block;
        if (corIdx < Dim)
        {
            STATE_T X2 = psireal[corIdx];
            STATE_T Y2 = psiimag[corIdx];
            psireal[realIdx] = 0;
            psireal[corIdx] = X2 * coef;
            psiimag[realIdx] = 0;
            psiimag[corIdx] = Y2 * coef;
        }
    }//checked
    
    /**************************************************************************************
    psireal:
    psiimag:
    pro:      save probability
    block:    qubit number
    m:        target qubit number
    dec:      target qubit state
    ****************************************************************************************/
    __global__ void  multiprob(
        STATE_T *psireal,
        STATE_T *psiimag,
        QSIZE Dim,
        STATE_T *pro, 
        QSIZE *block, 
        QSIZE m, 
        QSIZE dec)
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
            }                                                              //index 目标比特全为0
            QSIZE realIdx = index + dec;                                   //需要加的态的概率
            dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
            __syncthreads();//树状加法
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
            //计算时间,记录结果,只在 thread 0（即 threadIdx.x =
            //dprob[0]求和即得到最后的概率
            if (tid == 0)
            {
                pro[bid] = dprob[0];                       //再计算pro的和就得到最后的概率
            }
        }
    }

    __global__ void pmeasure_many_target(
        STATE_T* psireal,
        STATE_T* psiimag,
        STATE_T* result,
        QSIZE qnum_mask,
        QSIZE result_size,
        QSIZE Dim)
    {
        QSIZE bid = blockIdx.x;
        QSIZE tid = threadIdx.x;
        QSIZE idx = blockDim.x*bid + tid;    

        // 对每个result并行，并行个数为result_size个
        // 适用于target个数多于threaddim的情况，例如对10个qubit进行pmeasure
        result[idx] = 0;
        if (idx < result_size)
        {
            for (QSIZE i = 0; i < Dim / result_size; ++i)
            {
                QSIZE realIdx = 0;
                QSIZE copy_i = i;        // 这个i是要不断移位的，所以复制一份
                QSIZE copy_idx = idx;    // 同理

                // 下面计算realIdx
                // 例如：
                // qnum_mask : 00100100
                // copy_i = abcdef
                // copy_idx = xy
                //
                // realIdx 应该为 abxcdyef
                // 用不断右移的digit判断mask上是0还是1
                // 用flag判断一共有多少位：Dim=100000000，可以向右移动6次，如果是1，说明结束
                // 用set_digit说明正在操作哪一位，每次set_digit左移一位，realIdx = set_digit * (?) + realIdx
                // 如果digit & 1 == 0 说明是copy_i的序号，通过 copy_i & 1 取到最低位后，移动一位
                // 如果digit & 1 == 1 说明是copy_idx的序号，同理

                QSIZE set_digit = 1;
                QSIZE qnum_mask_copy = qnum_mask;

                int loops = 0;
                for (QSIZE flag = Dim;
                    flag != 1;
                    flag >>= 1)
                {    
                    loops++;
                    if ((qnum_mask_copy & 1) == 0)
                    {
                        realIdx += (set_digit *(copy_i & 1));
                        copy_i >>= 1;                                
                    }
                    else
                    {
                        realIdx += (set_digit *(copy_idx & 1));
                        copy_idx >>= 1;                        
                    }
                    set_digit <<= 1;
                    qnum_mask_copy >>= 1;
                }
                result[idx] += (
                    psireal[realIdx] * psireal[realIdx] + 
                    psiimag[realIdx] * psiimag[realIdx]
                    );
            }
        }
    }

    __global__ void pmeasure_one_target(
        STATE_T* psireal, 
        STATE_T* psiimag, 
        STATE_T* result,
        QSIZE qnum_mask,
        QSIZE result_idx,
        size_t result_dim,
        QSIZE Dim)
    {
        QSIZE idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        QSIZE bid = blockIdx.x, tid = threadIdx.x;

        extern __shared__ STATE_T dprob[];
        dprob[tid] = 0;
        
        if (idx < (Dim>>result_dim))
        {
            QSIZE copy_idx = idx;
            QSIZE copy_result_idx = result_idx;
            // 下面计算realIdx
            // 例如：
            // qnum_mask : 00100100
            // idx = abcdef
            // result_idx = xy
            //
            // realIdx 应该为 abxcdyef
            // 用不断右移的digit判断mask上是0还是1
            // 用flag判断一共有多少位：Dim=100000000，可以向右移动6次，如果是1，说明结束
            // 用set_digit说明正在操作哪一位，每次set_digit左移一位，realIdx = set_digit * (?) + realIdx
            // 如果digit & 1 == 0 说明是copy_idx的序号，通过 copy_idx & 1 取到最低位后，移动一位
            // 如果digit & 1 == 1 说明是copy_result_idx的序号，同理
            QSIZE realIdx = 0;
            QSIZE set_digit = 1;
            QSIZE qnum_mask_copy = qnum_mask;

            int loops = 0;
            for (QSIZE flag = Dim;
                flag != 1;
                flag >>= 1)
            {
                loops++;
                if ((qnum_mask_copy & 1) == 0)
                {
                    realIdx += (set_digit *(copy_idx & 1));
                    copy_idx >>= 1;                    
                }
                else
                {
                    realIdx += (set_digit *(copy_result_idx & 1));
                    copy_result_idx >>= 1;    
                }
                set_digit <<= 1;
                qnum_mask_copy >>= 1;
            }
            dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
                        
            __syncthreads();

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
            if (tid == 0)
            {
                result[bid] = dprob[0];
            }
        }
    }

    double randGenerator()
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

} // namespace gpu










