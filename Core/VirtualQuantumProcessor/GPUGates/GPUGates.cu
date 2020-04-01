/******************************************************************************
Copyright (c) 2017-2020 Origin Quantum Computing Co., Ltd.. All Rights Reserved.



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

#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <vector>
#include <algorithm>
#include <time.h>
#include "GPUGates.h"
using namespace std;

namespace gpu {
    __global__ void
        unitarysingle(
            gpu_qstate_t * psireal,
            gpu_qstate_t * psiimag,
            gpu_qsize_t Dim,
            gpu_qsize_t Block,
            gpu_qstate_t matrix_real00,
            gpu_qstate_t matrix_real01,
            gpu_qstate_t matrix_real10,
            gpu_qstate_t matrix_real11,
            gpu_qstate_t matrix_imag00,
            gpu_qstate_t matrix_imag01,
            gpu_qstate_t matrix_imag10,
            gpu_qstate_t matrix_imag11)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        gpu_qsize_t BlockNum = idx / Block;
        gpu_qsize_t BlockInt = idx % Block;
        gpu_qsize_t realIdx = BlockNum * 2 * Block + BlockInt;
        gpu_qsize_t corIdx = realIdx + Block;

        if (corIdx < Dim)
        {
            gpu_qstate_t X1 = psireal[realIdx];
            gpu_qstate_t X2 = psireal[corIdx];
            gpu_qstate_t Y1 = psiimag[realIdx];
            gpu_qstate_t Y2 = psiimag[corIdx];
            psireal[realIdx] = matrix_real00 * X1 - matrix_imag00 * Y1 + matrix_real01 * X2 - matrix_imag01 * Y2;
            psireal[corIdx] = matrix_real10 * X1 - matrix_imag10 * Y1 + matrix_real11 * X2 - matrix_imag11 * Y2;
            psiimag[realIdx] = matrix_real00 * Y1 + matrix_imag00 * X1 + matrix_real01 * Y2 + matrix_imag01 * X2;
            psiimag[corIdx] = matrix_real10 * Y1 + matrix_imag10 * X1 + matrix_real11 * Y2 + matrix_imag11 * X2;
        }
    }

    __global__ void controlunitarysingle(
        gpu_qstate_t * psireal,
        gpu_qstate_t * psiimag,
        gpu_qsize_t Dim,
        gpu_qsize_t target_qubit,
        gpu_qsize_t controller_mask,
        gpu_qstate_t matrix_real00,
        gpu_qstate_t matrix_real01,
        gpu_qstate_t matrix_real10,
        gpu_qstate_t matrix_real11,
        gpu_qstate_t matrix_imag00,
        gpu_qstate_t matrix_imag01,
        gpu_qstate_t matrix_imag10,
        gpu_qstate_t matrix_imag11
    )
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

        if (
            idx < Dim && 
            ((idx & controller_mask) == controller_mask) &&
            ((idx & target_qubit) == target_qubit)
           )
        {
            gpu_qsize_t corIdx = idx;                                     //1
            gpu_qsize_t realIdx = corIdx - target_qubit;                  //0
            gpu_qstate_t X1 = psireal[realIdx];
            gpu_qstate_t X2 = psireal[corIdx];
            gpu_qstate_t Y1 = psiimag[realIdx];
            gpu_qstate_t Y2 = psiimag[corIdx];
            psireal[realIdx] = matrix_real00 * X1 - matrix_imag00 * Y1 + matrix_real01 * X2 - matrix_imag01 * Y2;
            psireal[corIdx] = matrix_real10 * X1 - matrix_imag10 * Y1 + matrix_real11 * X2 - matrix_imag11 * Y2;
            psiimag[realIdx] = matrix_real00 * Y1 + matrix_imag00 * X1 + matrix_real01 * Y2 + matrix_imag01 * X2;
            psiimag[corIdx] = matrix_real10 * Y1 + matrix_imag10 * X1 + matrix_real11 * Y2 + matrix_imag11 * X2;
        }
    }

    __global__ void
        unitarydouble(
            gpu_qstate_t * psireal,
            gpu_qstate_t * psiimag,
            gpu_qsize_t Dim,
            gpu_qsize_t Block1,
            gpu_qsize_t Block2,
            gpu_qstate_t real0000,
            gpu_qstate_t real0001,
            gpu_qstate_t real0010,
            gpu_qstate_t real0011,
            gpu_qstate_t real0100,
            gpu_qstate_t real0101,
            gpu_qstate_t real0110,
            gpu_qstate_t real0111,
            gpu_qstate_t real1000,
            gpu_qstate_t real1001,
            gpu_qstate_t real1010,
            gpu_qstate_t real1011,
            gpu_qstate_t real1100,
            gpu_qstate_t real1101,
            gpu_qstate_t real1110,
            gpu_qstate_t real1111,
            gpu_qstate_t imag0000,
            gpu_qstate_t imag0001,
            gpu_qstate_t imag0010,
            gpu_qstate_t imag0011,
            gpu_qstate_t imag0100,
            gpu_qstate_t imag0101,
            gpu_qstate_t imag0110,
            gpu_qstate_t imag0111,
            gpu_qstate_t imag1000,
            gpu_qstate_t imag1001,
            gpu_qstate_t imag1010,
            gpu_qstate_t imag1011,
            gpu_qstate_t imag1100,
            gpu_qstate_t imag1101,
            gpu_qstate_t imag1110,
            gpu_qstate_t imag1111)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;
        gpu_qsize_t Idx00, Idx01, Idx10, Idx11;
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
            gpu_qstate_t X00 = psireal[Idx00];
            gpu_qstate_t X01 = psireal[Idx01];
            gpu_qstate_t X10 = psireal[Idx10];
            gpu_qstate_t X11 = psireal[Idx11];
            gpu_qstate_t Y00 = psiimag[Idx00];
            gpu_qstate_t Y01 = psiimag[Idx01];
            gpu_qstate_t Y10 = psiimag[Idx10];
            gpu_qstate_t Y11 = psiimag[Idx11];
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
        gpu_qstate_t * psireal,
        gpu_qstate_t * psiimag,
        gpu_qsize_t Dim,
        gpu_qsize_t controller_mask,
        gpu_qsize_t control_qubit,
        gpu_qsize_t target_qubit,
        gpu_qstate_t real0000,
        gpu_qstate_t real0001,
        gpu_qstate_t real0010,
        gpu_qstate_t real0011,
        gpu_qstate_t real0100,
        gpu_qstate_t real0101,
        gpu_qstate_t real0110,
        gpu_qstate_t real0111,
        gpu_qstate_t real1000,
        gpu_qstate_t real1001,
        gpu_qstate_t real1010,
        gpu_qstate_t real1011,
        gpu_qstate_t real1100,
        gpu_qstate_t real1101,
        gpu_qstate_t real1110,
        gpu_qstate_t real1111,
        gpu_qstate_t imag0000,
        gpu_qstate_t imag0001,
        gpu_qstate_t imag0010,
        gpu_qstate_t imag0011,
        gpu_qstate_t imag0100,
        gpu_qstate_t imag0101,
        gpu_qstate_t imag0110,
        gpu_qstate_t imag0111,
        gpu_qstate_t imag1000,
        gpu_qstate_t imag1001,
        gpu_qstate_t imag1010,
        gpu_qstate_t imag1011,
        gpu_qstate_t imag1100,
        gpu_qstate_t imag1101,
        gpu_qstate_t imag1110,
        gpu_qstate_t imag1111)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        if (
            idx < Dim &&
            ((idx & controller_mask) == controller_mask) &&
            ((idx & control_qubit) == control_qubit) &&
            ((idx & target_qubit) == target_qubit)
            )
        {
            gpu_qsize_t Idx00 = idx - control_qubit - target_qubit;
            gpu_qsize_t Idx01 = Idx00 - control_qubit;
            gpu_qsize_t Idx10 = Idx00 - target_qubit;
            gpu_qsize_t Idx11 = idx;
            gpu_qstate_t X00 = psireal[Idx00];
            gpu_qstate_t X01 = psireal[Idx01];
            gpu_qstate_t X10 = psireal[Idx10];
            gpu_qstate_t X11 = psireal[Idx11];
            gpu_qstate_t Y00 = psiimag[Idx00];
            gpu_qstate_t Y01 = psiimag[Idx01];
            gpu_qstate_t Y10 = psiimag[Idx10];
            gpu_qstate_t Y11 = psiimag[Idx11];
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

    __global__ void  initState(gpu_qstate_t * psireal, gpu_qstate_t * psiimag, gpu_qsize_t Dim)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number

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

    __global__ void qubitprob(gpu_qstate_t * psireal, gpu_qstate_t * psiimag,
                              gpu_qsize_t Dim, gpu_qsize_t Block, gpu_qstate_t *pr)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;
        gpu_qsize_t bid = blockIdx.x, tid = threadIdx.x;
        gpu_qsize_t BlockNum = idx / Block;
        gpu_qsize_t BlockInt = idx % Block;
        gpu_qsize_t realIdx = BlockNum * 2 * Block + BlockInt;
        gpu_qsize_t corIdx = realIdx + Block;
        extern __shared__ gpu_qstate_t  dprob[];
        dprob[tid] = 0;

        if (corIdx < Dim)
        {
            dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
            __syncthreads();
            gpu_qsize_t offset = 1, mask = 1;
            while (offset < kThreadDim)
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

    __global__ void probsumnew1(gpu_qstate_t * psireal, gpu_qstate_t * psiimag, gpu_qstate_t *probtemp, size_t num1, size_t m, size_t Dim, size_t * block)
    {
        size_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        size_t index1, index = 0, index2, k, s;
        gpu_qstate_t temp = 0;
        index1 = num1 + idx;                              //index1��ʾidx��Ӧ�Ĳ���Ȩ��̬����
        if (index1 < (1u << m))
        {
            for (size_t j = 0; j < m; j++)
            {
                index += block[j] * ((index1 >> j) % 2);
            }//index ��ʾidx��Ӧ��̬������
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

    __global__ void  probsum(gpu_qstate_t *pr, gpu_qstate_t *prob)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        if (0 == idx)
        {
            gpu_qstate_t dprob = 0;
            for (int i = 0; i < gridDim.x; i++)
            {
                dprob += pr[i];
            }
            *prob = dprob;
        }
    }//checked and can be optimized

    __global__ void  qubitcollapse0(gpu_qstate_t * psireal, gpu_qstate_t * psiimag, gpu_qsize_t Dim, gpu_qsize_t Block, gpu_qstate_t coef)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        gpu_qsize_t BlockNum = idx / Block;
        gpu_qsize_t BlockInt = idx % Block;
        gpu_qsize_t realIdx = BlockNum * 2 * Block + BlockInt;
        gpu_qsize_t corIdx = realIdx + Block;
        if (corIdx < Dim)
        {
            gpu_qstate_t X1 = psireal[realIdx];
            gpu_qstate_t Y1 = psiimag[realIdx];
            psireal[realIdx] = X1 * coef;
            psireal[corIdx] = 0;
            psiimag[realIdx] = Y1 * coef;
            psiimag[corIdx] = 0;
        }
    }//checked

    __global__ void  qubitcollapse1(gpu_qstate_t * psireal, gpu_qstate_t * psiimag, gpu_qsize_t Dim, gpu_qsize_t Block, gpu_qstate_t coef)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        gpu_qsize_t BlockNum = idx / Block;
        gpu_qsize_t BlockInt = idx % Block;
        gpu_qsize_t realIdx = BlockNum * 2 * Block + BlockInt;
        gpu_qsize_t corIdx = realIdx + Block;
        if (corIdx < Dim)
        {
            gpu_qstate_t X2 = psireal[corIdx];
            gpu_qstate_t Y2 = psiimag[corIdx];
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
    __global__ void  multiprob(gpu_qstate_t *psireal,
        gpu_qstate_t *psiimag,
        gpu_qsize_t Dim,
        gpu_qstate_t *pro,
        gpu_qsize_t *block,
        gpu_qsize_t m,
        gpu_qsize_t dec)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        gpu_qsize_t bid = blockIdx.x, tid = threadIdx.x;
        //gpu_qsize_t BlockNum = idx / Block;
        //gpu_qsize_t BlockInt = idx% Block;
        extern __shared__ gpu_qstate_t dprob[];
        dprob[tid] = 0;
        gpu_qsize_t i, j, k;
        if (idx < Dim / (1 << m))
        {
            gpu_qsize_t index = idx;
            for (i = 0; i < m; i++)
            {
                j = index / block[i];
                k = index % block[i];
                index = j * 2 * block[i] + k;
            }                                                              //index Ŀ������ȫΪ0
            gpu_qsize_t realIdx = index + dec;                                   //��Ҫ�ӵ�̬�ĸ���
            dprob[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
            __syncthreads();//��״�ӷ�
            int offset = 1, mask = 1;
            while (offset < kThreadDim)
            {
                if ((tid & mask) == 0)
                {
                    dprob[tid] += dprob[tid + offset];
                }
                offset += offset;
                mask = offset + mask;
                __syncthreads();
            }
            //����ʱ��,��¼����,ֻ�� thread 0���� threadIdx.x =
            //dprob[0]���ͼ��õ������ĸ���
            if (tid == 0)
            {
                pro[bid] = dprob[0];                       //�ټ���pro�ĺ;͵õ������ĸ���
            }
        }
    }

    __global__ void pmeasure_many_target(gpu_qstate_t* psireal,
        gpu_qstate_t* psiimag,
        double *result,
        gpu_qsize_t qnum_mask,
        gpu_qsize_t result_size,
        gpu_qsize_t Dim)
    {
        gpu_qsize_t bid = blockIdx.x;
        gpu_qsize_t tid = threadIdx.x;
        gpu_qsize_t idx = blockDim.x*bid + tid;

        // ��ÿ��result���У����и���Ϊresult_size��
        // ������target��������threaddim��������������10��qubit����pmeasure
        result[idx] = 0;
        if (idx < result_size)
        {
            for (gpu_qsize_t i = 0; i < Dim / result_size; ++i)
            {
                gpu_qsize_t realIdx = 0;
                gpu_qsize_t copy_i = i;        // ����i��Ҫ������λ�ģ����Ը���һ��
                gpu_qsize_t copy_idx = idx;    // ͬ��

                // ��������realIdx
                // ���磺
                // qnum_mask : 00100100
                // copy_i = abcdef
                // copy_idx = xy
                //
                // realIdx Ӧ��Ϊ abxcdyef
                // �ò������Ƶ�digit�ж�mask����0����1
                // ��flag�ж�һ���ж���λ��Dim=100000000�����������ƶ�6�Σ�������1��˵������
                // ��set_digit˵�����ڲ�����һλ��ÿ��set_digit����һλ��realIdx = set_digit * (?) + realIdx
                // ����digit & 1 == 0 ˵����copy_i�����ţ�ͨ�� copy_i & 1 ȡ������λ�����ƶ�һλ
                // ����digit & 1 == 1 ˵����copy_idx�����ţ�ͬ��

                gpu_qsize_t set_digit = 1;
                gpu_qsize_t qnum_mask_copy = qnum_mask;

                int loops = 0;
                for (gpu_qsize_t flag = Dim;
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

                result[idx] += psireal[realIdx] * psireal[realIdx] +
                               psiimag[realIdx] * psiimag[realIdx];
            }
        }
    }

    __global__ void pmeasure_one_target(
        gpu_qstate_t* psireal,
        gpu_qstate_t* psiimag,
        double *result,
        gpu_qsize_t qnum_mask,
        size_t result_idx,
        gpu_qsize_t result_dim,
        gpu_qsize_t Dim)
    {
        gpu_qsize_t idx = blockDim.x*blockIdx.x + threadIdx.x;           //thread number
        gpu_qsize_t bid = blockIdx.x, tid = threadIdx.x;

        extern __shared__ double dprob_result[];
        dprob_result[tid] = 0;
        
        if (idx < (Dim>>result_dim))
        {
            gpu_qsize_t copy_idx = idx;
            gpu_qsize_t copy_result_idx = result_idx;
            // ��������realIdx
            // ���磺
            // qnum_mask : 00100100
            // idx = abcdef
            // result_idx = xy
            //
            // realIdx Ӧ��Ϊ abxcdyef
            // �ò������Ƶ�digit�ж�mask����0����1
            // ��flag�ж�һ���ж���λ��Dim=100000000�����������ƶ�6�Σ�������1��˵������
            // ��set_digit˵�����ڲ�����һλ��ÿ��set_digit����һλ��realIdx = set_digit * (?) + realIdx
            // ����digit & 1 == 0 ˵����copy_idx�����ţ�ͨ�� copy_idx & 1 ȡ������λ�����ƶ�һλ
            // ����digit & 1 == 1 ˵����copy_result_idx�����ţ�ͬ��
            gpu_qsize_t realIdx = 0;
            gpu_qsize_t set_digit = 1;
            gpu_qsize_t qnum_mask_copy = qnum_mask;

            int loops = 0;
            for (gpu_qsize_t flag = Dim;
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
            dprob_result[tid] = psireal[realIdx] * psireal[realIdx] + psiimag[realIdx] * psiimag[realIdx];
                        
            __syncthreads();

            size_t offset = 1, mask = 1;
            while (offset < kThreadDim)
            {
                if ((tid & mask) == 0)
                {
                    dprob_result[tid] += dprob_result[tid + offset];
                }
                offset += offset;
                mask = offset + mask;
                __syncthreads();
            }

            if (tid == 0)
            {
                result[bid] = dprob_result[0];
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










