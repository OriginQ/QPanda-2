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
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"


#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


namespace gpu{

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
    gpu_qstate_t matrix_imag11);

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
);

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
    gpu_qstate_t imag1111);

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
    gpu_qstate_t imag1111);

__global__ void initState(gpu_qstate_t * psireal, gpu_qstate_t * psiimag, gpu_qsize_t Dim);

__global__ void qubitprob(gpu_qstate_t * psireal,
    gpu_qstate_t * psiimag,
    gpu_qsize_t Dim,
    gpu_qsize_t Block,
    gpu_qstate_t *pr);

__global__ void probsum(gpu_qstate_t * pr, gpu_qstate_t * prob);
__global__ void qubitcollapse0(
    gpu_qstate_t * psireal,
    gpu_qstate_t * psiimag,
    gpu_qsize_t Dim,
    gpu_qsize_t Block,
    gpu_qstate_t coef);
__global__ void qubitcollapse1(
    gpu_qstate_t * psireal,
    gpu_qstate_t * psiimag,
    gpu_qsize_t Dim,
    gpu_qsize_t Block,
    gpu_qstate_t coef);
__global__ void multiprob(
    gpu_qstate_t * psireal,
    gpu_qstate_t * psiimag,
    gpu_qsize_t Dim,
    gpu_qstate_t * pro,
    gpu_qsize_t * block,
    gpu_qsize_t m,
    gpu_qsize_t dec);
__global__ void probsumnew1(
    gpu_qstate_t * psireal,
    gpu_qstate_t * psiimag,
    gpu_qstate_t *probtemp,
    size_t num1,
    size_t m,
    size_t Dim,
    size_t * block);

__global__ void pmeasure_many_target(
    gpu_qstate_t* psireal,
    gpu_qstate_t* psiimag,
    double* result,
    gpu_qsize_t qnum_mask,
    gpu_qsize_t result_size,
    gpu_qsize_t Dim);

__global__ void pmeasure_one_target(
    gpu_qstate_t* psireal,
    gpu_qstate_t* psiimag,
    double* result,
    gpu_qsize_t qnum_mask,
    size_t result_idx,
    gpu_qsize_t result_dim,
    gpu_qsize_t Dim);

double randGenerator();

}//namespace gpu

#endif // !_GPU_GATES_DECL_H

