
/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Definition of Encapsulation of GPU gates
************************************************************************/

#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "GPUGatesWrapper.h"
#include "GPUGates.h"


using namespace std;
#define SET_BLOCKDIM  BLOCKDIM = (1ull << (psigpu.qnum - 1)) / kThreadDim;

static bool pMeasure_few_target(GATEGPU::QState&, vector<double>&result, Qnum&);
static bool pMeasure_many_target(GATEGPU::QState&, vector<double>&result, Qnum&);

static gpu_qsize_t getControllerMask(Qnum& qnum, int target = 1)
{
    gpu_qsize_t qnum_mask = 0;

    // obtain the mask for controller qubit
    for (auto iter = qnum.begin(); iter != qnum.end() - target; ++iter)
    {
        qnum_mask += (1ull << *iter);
    }
    return qnum_mask;
}

int GATEGPU::devicecount()
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

static bool getSynchronizeResult( cudaError_t cudaStatue, const char * pcGate)
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

bool GATEGPU::destroyState(QState& psi, QState& psigpu, size_t stQnum)
{
    if ((nullptr == psi.real) ||
        (nullptr == psi.imag) ||
        (nullptr == psigpu.real) ||
        (nullptr == psigpu.imag))
    {
        return false;
    }

    if (stQnum < 30)
    {
        cudaError_t cuda_status = cudaFree(psigpu.real);
        if (cudaSuccess != cuda_status)
        {
            cout << "psigpu.real free error" << endl;
            return false;
        }
        cuda_status = cudaFree(psigpu.imag);
        if (cudaSuccess != cuda_status)
        {
            cout << "psigpu.imag free error" << endl;
            return false;
        }
        free(psi.real);
        free(psi.imag);
        psi.real = nullptr;
        psi.imag = nullptr;
        psigpu.real = nullptr;
        psigpu.imag = nullptr;
    }
    else
    {
        cudaFreeHost(psigpu.real);
        cudaFreeHost(psigpu.imag);
        psigpu.real = nullptr;
        psigpu.imag = nullptr;
    }

    return true;
}

bool GATEGPU::clearState(QState& psi, QState& psigpu, size_t stQnum)
{
    if ((nullptr == psi.real) ||
        (nullptr == psi.imag) ||
        (nullptr == psigpu.real) ||
        (nullptr == psigpu.imag))
    {
        return false;
    }

    if (stQnum < 30)
    {
        gpu_qsize_t BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / kThreadDim;
        gpu::initState<<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >>>(psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }
    else
    {
        gpu_qsize_t BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / kThreadDim;
        gpu::initState <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >>> (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }

    return true;
}

bool GATEGPU::initstate(QState& psi, QState& psigpu, size_t qnum)
{
    if (qnum >= 30)
    {
        if (nullptr == psi.real)
        {
            cudaError_t cuda_status = cudaHostAlloc(&psi.real, sizeof(gpu_qstate_t)*(1ll << qnum), cudaHostAllocMapped);
            if (cuda_status != cudaSuccess)
            {
                printf("host alloc fail!\n");
                return false;
            }
            cudaHostGetDevicePointer(&psigpu.real, psi.real, 0);
        }

        if (nullptr == psi.imag)
        {
            cudaError_t cuda_status1 = cudaHostAlloc(&psi.imag, sizeof(gpu_qstate_t)*(1ll << qnum), cudaHostAllocMapped);
            if (cuda_status1 != cudaSuccess)
            {
                printf("host alloc fail!\n");
                cudaFreeHost(psi.real);
                return false;
            }
            cudaHostGetDevicePointer(&psigpu.imag, psi.imag, 0);
        }
        
        psi.qnum = qnum;
        psigpu.qnum = qnum;
        gpu_qsize_t BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / kThreadDim;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }
    else
    {
        gpu_qsize_t Dim = 1ull << qnum;
        cudaError_t cuda_status;

        if (nullptr == psi.real)
        {
            psi.real = (gpu_qstate_t*)malloc(sizeof(gpu_qstate_t)*Dim);
        }

        if (nullptr == psi.imag)
        {
            psi.imag = (gpu_qstate_t*)malloc(sizeof(gpu_qstate_t)*Dim);
        }
        
        if (nullptr == psigpu.real)
        {
            cuda_status = cudaMalloc((void**)&psigpu.real, sizeof(gpu_qstate_t)*Dim);
            if (cudaSuccess != cuda_status)
            {
                printf("psigpu.real alloc gpu memoery error!\n");
                free(psi.real);
                free(psi.imag);
                return false;
            }
        }

        if (nullptr == psigpu.imag)
        {
            cuda_status = cudaMalloc((void**)&psigpu.imag, sizeof(gpu_qstate_t)*Dim);
            if (cudaSuccess != cuda_status)
            {
                printf("psigpu.imag alloc gpu memoery error!\n");
                free(psi.real);
                free(psi.imag);
                cudaFree(psigpu.real);
                return false;
            }
        }
        psigpu.qnum = qnum;
        psi.qnum = qnum;
        gpu_qsize_t BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / kThreadDim;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));

    }
    return true;
}

bool GATEGPU::unitarysingle(
    QState& psigpu,
    size_t qn,
    QState& matrix,
    bool isConjugate,
    double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {

        if (isConjugate)
        {
            gpu_qstate_t temp_real, temp_imag;
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

        gpu_qstate_t real00 = matrix.real[0];
        gpu_qstate_t real01 = matrix.real[1];
        gpu_qstate_t real10 = matrix.real[2];
        gpu_qstate_t real11 = matrix.real[3];
        gpu_qstate_t imag00 = matrix.imag[0];
        gpu_qstate_t imag01 = matrix.imag[1];
        gpu_qstate_t imag10 = matrix.imag[2];
        gpu_qstate_t imag11 = matrix.imag[3];

        //test

        gpu_qsize_t BLOCKDIM;
        SET_BLOCKDIM
            gpu::unitarysingle << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn, real00, real01, real10, real11, imag00, imag01, imag10, imag11);

        return true;
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
        if (isConjugate)
        {
            gpu_qstate_t temp_real, temp_imag;
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
        gpu_qsize_t target_qubit = 1ull << qnum.back();

        // 1 is for the control single gate
        gpu_qsize_t mask = getControllerMask(qnum, 1);

        gpu_qstate_t real00 = matrix.real[0];
        gpu_qstate_t real01 = matrix.real[1];
        gpu_qstate_t real10 = matrix.real[2];
        gpu_qstate_t real11 = matrix.real[3];
        gpu_qstate_t imag00 = matrix.imag[0];
        gpu_qstate_t imag01 = matrix.imag[1];
        gpu_qstate_t imag10 = matrix.imag[2];
        gpu_qstate_t imag11 = matrix.imag[3];

        gpu_qsize_t BLOCKDIM;
        SET_BLOCKDIM;

        BLOCKDIM = (1ull << (psigpu.qnum)) / kThreadDim;

        gpu::controlunitarysingle << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
            (
                psigpu.real, 
                psigpu.imag, 
                1ull << (psigpu.qnum), 
                target_qubit,
                mask,                 
                real00, real01, real10, real11, imag00, imag01, imag10, imag11
            );

        return true;
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
            gpu_qstate_t temp_real, temp_imag;
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
                //matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        gpu_qsize_t BLOCKDIM;
        SET_BLOCKDIM
            gpu::unitarydouble << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn_0, 1ull << qn_1,
                matrix.real[0], matrix.real[1], matrix.real[2], matrix.real[3],
                matrix.real[4], matrix.real[5], matrix.real[6], matrix.real[7],
                matrix.real[8], matrix.real[9], matrix.real[10], matrix.real[11],
                matrix.real[12], matrix.real[13], matrix.real[14], matrix.real[15],
                matrix.imag[0], matrix.imag[1], matrix.imag[2], matrix.imag[3],
                matrix.imag[4], matrix.imag[5], matrix.imag[6], matrix.imag[7],
                matrix.imag[8], matrix.imag[9], matrix.imag[10], matrix.imag[11],
                matrix.imag[12], matrix.imag[13], matrix.imag[14], matrix.imag[15]);

        return true;
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
        if (isConjugate)
        {
            gpu_qstate_t temp_real, temp_imag;
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
        gpu_qsize_t m = qnum.size();
        gpu_qsize_t control_qubit = qnum[m - 2];
        gpu_qsize_t target_qubit = qnum.back();
        //sort(qnum.begin(), qnum.end());
        gpu_qsize_t mask = getControllerMask(qnum, 1);
        gpu_qsize_t BLOCKDIM;
        SET_BLOCKDIM
            gpu::controlunitarydouble << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), mask, 1ull << control_qubit, 1ull << target_qubit,
                matrix.real[0], matrix.real[1], matrix.real[2], matrix.real[3],
                matrix.real[4], matrix.real[5], matrix.real[6], matrix.real[7],
                matrix.real[8], matrix.real[9], matrix.real[10], matrix.real[11],
                matrix.real[12], matrix.real[13], matrix.real[14], matrix.real[15],
                matrix.imag[0], matrix.imag[1], matrix.imag[2], matrix.imag[3],
                matrix.imag[4], matrix.imag[5], matrix.imag[6], matrix.imag[7],
                matrix.imag[8], matrix.imag[9], matrix.imag[10], matrix.imag[11],
                matrix.imag[12], matrix.imag[13], matrix.imag[14], matrix.imag[15]);
        return true;
    }

    return true;
}

//bool GATEGPU::DiagonalGate(
//    QState& psigpu,
//    Qnum& qnum,
//    QState& matrix,
//    bool isConjugate,
//    double error_rate)
//{
//    if (gpu::randGenerator() > error_rate)
//    {
//        if (isConjugate)
//        {
//            for (size_t i = 0; i < (1 << qnum); i++)
//            {
//                matrix.image[i] = -matrix.imag[i];
//            }
//        }
//        gpu_qsize_t m = qnum.size();
//        gpu_qsize_t mask = getControllerMask(qnum, 1);
//        gpu_qsize_t BLOCKDIM;
//        SET_BLOCKDIM
//            gpu::DiagonalGate << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
//            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), mask, matrix);
//        return true;
//    }
//
//    return true;
//}






//qbReset
bool GATEGPU::qbReset(QState& psigpu, size_t qn, double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        cudaError_t cuda_status;
        gpu_qstate_t * resultgpu;
        // cudaHostAlloc((void **)&result, sizeof(gpu_qstate_t)*(psigpu.qnum-1))/kThreadDim, cudaHostAllocMapped);
        //cudaHostGetDevicePointer(&resultgpu, result, 0);

        cuda_status = cudaMalloc((void **)&resultgpu, sizeof(gpu_qstate_t)*(1ull << (psigpu.qnum - 1)) / kThreadDim);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMalloc error\n");
            return false;
        }

        gpu_qstate_t * probgpu, *prob;
        cuda_status = cudaHostAlloc((void **)&prob, sizeof(gpu_qstate_t), cudaHostAllocMapped);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaHostAlloc error\n");
            cudaFree(resultgpu);
        }
        cudaHostGetDevicePointer(&probgpu, prob, 0);

        gpu_qsize_t BLOCKDIM;
        SET_BLOCKDIM
            gpu::qubitprob << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim, kThreadDim * sizeof(gpu_qstate_t) >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn, resultgpu);
        cuda_status = cudaDeviceSynchronize();
        gpu::probsum << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> > (resultgpu, probgpu);
        cuda_status = cudaDeviceSynchronize();
        *prob = 1 / sqrt(*prob);
        gpu::qubitcollapse0 << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn, *prob);
        cuda_status = cudaDeviceSynchronize();

        cudaFree(resultgpu);
        cudaFreeHost(prob);
        return getSynchronizeResult(cuda_status, "qReset");
    }

    return true;
}

int GATEGPU::qubitmeasure(QState& psigpu, gpu_qsize_t Block, gpu_qstate_t* resultgpu, gpu_qstate_t* probgpu)
{
    //double * resultgpu;
    gpu_qsize_t BLOCKDIM;
    SET_BLOCKDIM

    gpu_qsize_t count = (0 == BLOCKDIM) ? 1 : BLOCKDIM;
    gpu_qstate_t prob;
    cudaError_t cuda_status;
    if (nullptr == resultgpu)
    {
        cuda_status = cudaMalloc(&resultgpu, sizeof(gpu_qstate_t) * count);
        if (cudaSuccess != cuda_status)
        {
            cout << "resultgpu  " << cudaGetErrorString(cuda_status) << endl;
            return -1;
        }
    }

    if (nullptr == probgpu)
    {
        cuda_status = cudaMalloc(&probgpu, sizeof(gpu_qstate_t));
        if (cudaSuccess != cuda_status)
        {
            cout << "probgpu  " << cudaGetErrorString(cuda_status) << endl;
            cudaFree(resultgpu);
            resultgpu = nullptr;
            return -1;
        }
    }

    gpu::qubitprob << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim, kThreadDim * sizeof(gpu_qstate_t) >> >
        (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, resultgpu);
    cuda_status = cudaDeviceSynchronize(); 
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }

    gpu::probsum << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >> > (resultgpu, probgpu);
    cuda_status = cudaDeviceSynchronize();
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }

    cuda_status = cudaMemcpy(&prob, probgpu, sizeof(gpu_qstate_t), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_status)
    {
        fprintf(stderr, "cudaMemcpy error\n");
        return -1;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }

    int outcome = 0;
    if (gpu::randGenerator() > prob)
    {
        outcome = 1;
    }

    if (0 == outcome)
    {
        if (prob < 0.000001 && prob > -0.000001)
        {
            return outcome;
        }

        prob = 1 / sqrt(prob);
        gpu::qubitcollapse0 <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >>>
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, prob);
    }
    else
    {
        if (1 - prob < 0.000001 && 1- prob > -0.000001)
        {
            return outcome;
        }

        prob = 1 / sqrt(1 - prob);
        gpu::qubitcollapse1 <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)kThreadDim >>>
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, prob);
    }
    cuda_status = cudaDeviceSynchronize();
    getSynchronizeResult(cuda_status, "qubitmeasure");

    return outcome;
}//checked

bool probcompare(pair<size_t, gpu_qstate_t>& a, pair<size_t, gpu_qstate_t>& b)
{
    return a.second > b.second;
}

bool GATEGPU::pMeasurenew(QState& psigpu,
    touple_prob &vprob,
    Qnum& qnum,
    int select_max)
{
	// 10 可能是一个比较好的阈值，因为1024为线程单位，就更适合使用many_target
    vec_prob result;
	bool status = false;

	if (qnum.size() < 10)
	{
        status = pMeasure_few_target(psigpu, result, qnum);
	}
	else
	{
        status = pMeasure_many_target(psigpu, result, qnum);
	}

	if (status)
	{
		size_t i = 0;
        for (auto &aiter : result)
		{
			vprob.push_back(make_pair(i, aiter));
			i++;
		}
	}
    else
    {
        throw std::runtime_error("PMeasure error");
    }

	return status;
	
}

bool GATEGPU::pMeasure_no_index(
    QState& psigpu,
    vec_prob &mResult,
    Qnum& qnum)
{
    // 10 可能是一个比较好的阈值，因为1024为线程单位，就更适合使用many_target

    bool status = false;
    if (qnum.size() < 10)
    {
        status = pMeasure_few_target(psigpu, mResult, qnum);
    }
    else
    {
        status = pMeasure_many_target(psigpu, mResult, qnum);
    }

    if (!status)
    {
        throw std::runtime_error("pmeasure error");
    }
    return true;
}

static bool pMeasure_few_target(GATEGPU::QState& psigpu, vector<double> &result, Qnum& qnum)
{
    gpu_qsize_t result_size = 1ull << qnum.size();
    result.assign(result_size, 0);

    gpu_qsize_t Dim = 1ull << psigpu.qnum; // 态矢总长度
    gpu_qsize_t BLOCKDIM;
    BLOCKDIM = Dim / result_size;

    cudaError_t cudaStatus;
    // 一般来说BLOCKDIM不可能为0，因为对于few target的情况，result_size<10
    // 保险起见
    BLOCKDIM = (BLOCKDIM == 0 ? 1 : BLOCKDIM);

    double* result_gpu = nullptr;
    cudaStatus = cudaMalloc(&result_gpu, sizeof(double)*BLOCKDIM);
    if (cudaSuccess != cudaStatus)
    {
        return false;
    }

    double *result_cpu = (double *)malloc(sizeof(double) * BLOCKDIM);
    if (nullptr == result_cpu)
    {
        cudaFree(result_gpu);
        return false;
    }

    gpu_qsize_t qnum_mask = 0;
    // obtain the mask for pMeasure qubit
    for (auto iter : qnum)
    {
        qnum_mask += (1ull << iter);
    }
    gpu_qsize_t SHARED_SIZE = kThreadDim * sizeof(double);
    for (size_t result_idx = 0; result_idx < result_size; ++result_idx)
    {
        gpu::pmeasure_one_target << < BLOCKDIM, kThreadDim, SHARED_SIZE >> > (
            psigpu.real,
            psigpu.imag,
            result_gpu,
            qnum_mask,
            result_idx,
            qnum.size(),
            Dim);

        cudaStatus = cudaMemcpy(result_cpu, result_gpu, sizeof(double)*BLOCKDIM, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cudaStatus)
        {
            cudaFree(result_gpu);
            free(result_cpu);
            return false;
        }
        
        double result_sum = 0;
        for (size_t i = 0; i < BLOCKDIM; ++i)
        {
            result_sum += result_cpu[i];
        }            
        result[result_idx] = result_sum;
    }

    cudaFree(result_gpu);
    free(result_cpu);

    return true;
}

static bool pMeasure_many_target(GATEGPU::QState& psigpu, vector<double> &result, Qnum& qnum)
{
    gpu_qsize_t qnum_mask = 0;

    // obtain the mask for pMeasure qubit
    for (auto iter : qnum)
    {
        qnum_mask += (1ull << iter);
    }

    gpu_qsize_t result_size = 1ull << qnum.size();
    result.resize(result_size);

    // allocate the graphics memory for result
    double* result_gpu = nullptr;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&result_gpu, result_size * sizeof(double));
    if (cudaSuccess != cudaStatus)
    {
        return false;
    }
    cudaMemset(result_gpu, 0, result_size * sizeof(double));

    gpu_qsize_t BLOCKDIM;
    BLOCKDIM = result_size / kThreadDim;

    gpu::pmeasure_many_target <<< (BLOCKDIM == 0 ? 1 : BLOCKDIM), kThreadDim >>> (
        psigpu.real,
        psigpu.imag,
        result_gpu,
        qnum_mask,
        result_size,
        1ull << (psigpu.qnum));

    cudaMemcpy(result.data(), result_gpu, result_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(result_gpu);
    return true;
}

bool GATEGPU::getState(QState &psi, QState &psigpu, size_t qnum)
{
    cudaError_t cuda_status;

    if (qnum < 30)
    {
        gpu_qsize_t Dim = 1ull << qnum;
        cuda_status = cudaMemcpy(psi.real, psigpu.real, sizeof(gpu_qstate_t)*Dim, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            return false;
        }

        cuda_status = cudaMemcpy(psi.imag, psigpu.imag, sizeof(gpu_qstate_t)*Dim, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status)
        {
            fprintf(stderr, "cudaMemcpy error\n");
            return false;
        }
    }
    return true;
}

void GATEGPU::gpuFree(void* memory)
{
    if (memory != nullptr)
    {
        cudaFree(memory);
    }
}
