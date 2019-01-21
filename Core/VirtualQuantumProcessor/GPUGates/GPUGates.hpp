/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Cuda function of quantum gates, defined in GPUGates.cu
************************************************************************/

#ifndef _GPU_GATES_DECL_H
#define _GPU_GATES_DECL_H

#include <string>
#include <math.h>
#include <vector>    
#include <iostream>
#include "GPUStruct.hpp"

namespace gpu{

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
    STATE_T matrix_imag11);

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
);

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
    STATE_T imag1111);

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
    STATE_T imag1111);

__global__ void initState(STATE_T * psireal, STATE_T * psiimag, QSIZE Dim);

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
__global__ void probsumnew1(
    STATE_T * psireal,
    STATE_T * psiimag,
    STATE_T *probtemp,
    size_t num1,
    size_t m,
    size_t Dim,
    size_t * block);

__global__ void pmeasure_many_target(
    STATE_T* psireal,
    STATE_T* psiimag,
    STATE_T* result,
    QSIZE qnum_mask,
    QSIZE result_size,
    QSIZE Dim);

__global__ void pmeasure_one_target(
    STATE_T* psireal,
    STATE_T* psiimag,
    STATE_T* result,
    QSIZE qnum_mask,
    QSIZE result_idx,
    size_t result_dim,
    QSIZE Dim);

double randGenerator();

}//namespace gpu

#endif // !_GPU_GATES_DECL_H

