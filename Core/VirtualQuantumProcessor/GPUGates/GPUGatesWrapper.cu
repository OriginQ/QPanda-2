
/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Definition of Encapsulation of GPU gates
************************************************************************/

#include "GPUGatesWrapper.hpp"
#include "GPUGates.hpp"

using namespace std;

#define SET_BLOCKDIM  BLOCKDIM = (1ull << (psigpu.qnum - 1)) / THREADDIM;

static QSIZE getControllerMask(GATEGPU::Qnum& qnum, int target = 1)
{
    QSIZE qnum_mask = 0;

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
        QSIZE BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / THREADDIM;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }
    else
    {
        QSIZE BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / THREADDIM;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }

    return true;
}

bool GATEGPU::initstate(QState& psi, QState& psigpu, size_t qnum)
{
    if (qnum >= 30)
    {
        if (nullptr == psi.real)
        {
            cudaError_t cuda_status = cudaHostAlloc(&psi.real, sizeof(double)*(1ll << qnum), cudaHostAllocMapped);
            if (cuda_status != cudaSuccess)
            {
                printf("host alloc fail!\n");
                return false;
            }
            cudaHostGetDevicePointer(&psigpu.real, psi.real, 0);
        }

        if (nullptr == psi.imag)
        {
            cudaError_t cuda_status1 = cudaHostAlloc(&psi.imag, sizeof(double)*(1ll << qnum), cudaHostAllocMapped);
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
        QSIZE BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / THREADDIM;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));
    }
    else
    {
        QSIZE Dim = 1ull << qnum;
        cudaError_t cuda_status;

        if (nullptr == psi.real)
        {
            psi.real = (STATE_T*)malloc(sizeof(STATE_T)*Dim);
        }

        if (nullptr == psi.imag)
        {
            psi.imag = (STATE_T*)malloc(sizeof(STATE_T)*Dim);
        }
        
        if (nullptr == psigpu.real)
        {
            cuda_status = cudaMalloc((void**)&psigpu.real, sizeof(STATE_T)*Dim);
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
            cuda_status = cudaMalloc((void**)&psigpu.imag, sizeof(STATE_T)*Dim);
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
        QSIZE BLOCKDIM;
        BLOCKDIM = (1ull << psigpu.qnum) / THREADDIM;
        gpu::initState << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum));

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

        double real00 = matrix.real[0];
        double real01 = matrix.real[1];
        double real10 = matrix.real[2];
        double real11 = matrix.real[3];
        double imag00 = matrix.imag[0];
        double imag01 = matrix.imag[1];
        double imag10 = matrix.imag[2];
        double imag11 = matrix.imag[3];

        //test

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
            gpu::unitarysingle << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> >
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
        QSIZE target_qubit = 1ull << qnum.back();

        // 1 is for the control single gate
        QSIZE mask = getControllerMask(qnum, 1);         

        double real00 = matrix.real[0];
        double real01 = matrix.real[1];
        double real10 = matrix.real[2];
        double real11 = matrix.real[3];
        double imag00 = matrix.imag[0];
        double imag01 = matrix.imag[1];
        double imag10 = matrix.imag[2];
        double imag11 = matrix.imag[3];

        QSIZE BLOCKDIM;
        SET_BLOCKDIM;

        BLOCKDIM = (1ull << (psigpu.qnum)) / THREADDIM;

        gpu::controlunitarysingle << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> >
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
                //matrix[i].imag = -matrix[i].imag;
                matrix.real[i] = matrix.real[i];
                matrix.imag[i] = -matrix.imag[i];
                //matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
            }//dagger
        }

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
            gpu::unitarydouble << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> >
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
        QSIZE control_qubit = qnum[m - 2];
        QSIZE target_qubit = qnum.back();
        //sort(qnum.begin(), qnum.end());
        QSIZE mask = getControllerMask(qnum, 1);
        QSIZE BLOCKDIM;
        SET_BLOCKDIM
            gpu::controlunitarydouble << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> >
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

//qbReset
bool GATEGPU::qbReset(QState& psigpu, size_t qn, double error_rate)
{
    if (gpu::randGenerator() > error_rate)
    {
        cudaError_t cuda_status;
        double * resultgpu;
        // cudaHostAlloc((void **)&result, sizeof(STATE_T)*(psigpu.qnum-1))/THREADDIM, cudaHostAllocMapped);
        //cudaHostGetDevicePointer(&resultgpu, result, 0);

        cuda_status = cudaMalloc((void **)&resultgpu, sizeof(STATE_T)*(1ull << (psigpu.qnum - 1)) / THREADDIM);
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
        cudaHostGetDevicePointer(&probgpu, prob, 0);

        QSIZE BLOCKDIM;
        SET_BLOCKDIM
            gpu::qubitprob << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM, THREADDIM * sizeof(STATE_T) >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn, resultgpu);
        cuda_status = cudaDeviceSynchronize();
        gpu::probsum << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (resultgpu, probgpu);
        cuda_status = cudaDeviceSynchronize();
        *prob = 1 / sqrt(*prob);
        gpu::qubitcollapse0 << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> >
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), 1ull << qn, *prob);
        cuda_status = cudaDeviceSynchronize();

        cudaFree(resultgpu);
        cudaFreeHost(prob);
        return getSynchronizeResult(cuda_status, "qReset");
    }

    return true;
}

int GATEGPU::qubitmeasure(QState& psigpu, QSIZE Block, double* &resultgpu, double* &probgpu)
{
    //double * resultgpu;
    QSIZE BLOCKDIM;
    SET_BLOCKDIM

    QSIZE count = (0 == BLOCKDIM) ? 1 : BLOCKDIM;
    double prob;
    cudaError_t cuda_status;
    if (nullptr == resultgpu)
    {
        cuda_status = cudaMalloc(&resultgpu, sizeof(STATE_T)* count);
        if (cudaSuccess != cuda_status)
        {
            cout << "resultgpu  " << cudaGetErrorString(cuda_status) << endl;
            return -1;
        }
    }

    if (nullptr == probgpu)
    {
        cuda_status = cudaMalloc(&probgpu, sizeof(STATE_T));
        if (cudaSuccess != cuda_status)
        {
            cout << "probgpu  " << cudaGetErrorString(cuda_status) << endl;
            cudaFree(resultgpu);
            resultgpu = nullptr;
            return -1;
        }
    }

    gpu::qubitprob << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM, THREADDIM * sizeof(STATE_T) >> >
        (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, resultgpu);
    cuda_status = cudaDeviceSynchronize(); 
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }

    gpu::probsum << < (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >> > (resultgpu, probgpu); 
    cuda_status = cudaDeviceSynchronize();
    if (cudaSuccess != cuda_status)
    {
        cout << cudaGetErrorString(cuda_status) << endl;
        return -1;
    }

    cuda_status = cudaMemcpy(&prob, probgpu, sizeof(STATE_T), cudaMemcpyDeviceToHost);
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
        prob = 1 / sqrt(prob);
        gpu::qubitcollapse0 <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >>>
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, prob);
    }
    else
    {
        prob = 1 / sqrt(1 - prob);
        gpu::qubitcollapse1 <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >>>
            (psigpu.real, psigpu.imag, 1ull << (psigpu.qnum), Block, prob);
    }
    cuda_status = cudaDeviceSynchronize();
    getSynchronizeResult(cuda_status, "qubitmeasure");

    return outcome;
}//checked

bool probcompare(pair<size_t, double>& a, pair<size_t, double>& b)
{
    return a.second > b.second;
}

bool GATEGPU::pMeasurenew(
    QState& psigpu,
    vector<pair<size_t, double>>& vprob,
    Qnum& qnum,
    int select_max)
{
    return false;
}

static bool pMeasure_few_target(GATEGPU::QState&, vector<double>&, GATEGPU::Qnum&);
static bool pMeasure_many_target(GATEGPU::QState&, vector<double>&, GATEGPU::Qnum&);

bool GATEGPU::pMeasure_no_index(
    QState& psigpu,
    vector<double> &mResult,
    Qnum& qnum)
{
    // 10 可能是一个比较好的阈值，因为1024为线程单位，就更适合使用many_target
    if (qnum.size() < 10)
    {
        return pMeasure_few_target(psigpu, mResult, qnum);
    }
    else
    {
        return pMeasure_many_target(psigpu, mResult, qnum);        
    }
}

static bool pMeasure_few_target(GATEGPU::QState& psigpu, vector<double>& mResult, GATEGPU::Qnum& qnum)
{
    QSIZE result_size = 1ull << qnum.size();
    mResult.resize(result_size);

    QSIZE Dim = 1ull << psigpu.qnum; // 态矢总长度
    QSIZE BLOCKDIM;
    BLOCKDIM = Dim / result_size;

    cudaError_t cudaStatus;
    // 一般来说BLOCKDIM不可能为0，因为对于few target的情况，result_size<10
    // 保险起见
    BLOCKDIM = (BLOCKDIM == 0 ? 1 : BLOCKDIM);

    STATE_T* result_gpu;
    cudaStatus = cudaMalloc(&result_gpu, sizeof(STATE_T)*BLOCKDIM);

    STATE_T* result_cpu;
    result_cpu = (STATE_T*)malloc(sizeof(STATE_T)*BLOCKDIM);

    QSIZE qnum_mask = 0;
    // obtain the mask for pMeasure qubit
    for (auto iter : qnum)
    {
        qnum_mask += (1ull << iter);
    }
    QSIZE SHARED_SIZE = THREADDIM * sizeof(STATE_T);
    for (int result_idx = 0; result_idx < result_size; ++result_idx)
    {
        gpu::pmeasure_one_target << < (unsigned)BLOCKDIM, (unsigned)THREADDIM, (unsigned)SHARED_SIZE >> > (
            psigpu.real,
            psigpu.imag,
            result_gpu,
            qnum_mask,
            result_idx,
            qnum.size(),
            Dim);

        cudaStatus = cudaMemcpy(result_cpu, result_gpu, sizeof(STATE_T)*BLOCKDIM, cudaMemcpyDeviceToHost);
        
        STATE_T result_sum = 0;
        for (int i = 0; i < BLOCKDIM; ++i)
        {
            result_sum += result_cpu[i];
        }            
        mResult[result_idx] = result_sum;
    }
    cudaFree(result_gpu);
    free(result_cpu);

    return true;
}

static bool pMeasure_many_target(GATEGPU::QState& psigpu, vector<double>& mResult, GATEGPU::Qnum& qnum)
{
    QSIZE qnum_mask = 0;

    // obtain the mask for pMeasure qubit
    for (auto iter : qnum)
    {
        qnum_mask += (1ull << iter);
    }

    QSIZE result_size = 1ull << qnum.size();
    mResult.resize(result_size);

    // allocate the graphics memory for result
    STATE_T* result_gpu;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&result_gpu, result_size * sizeof(STATE_T));
    cudaStatus = cudaMemset(result_gpu, 0, result_size * sizeof(STATE_T));

    QSIZE BLOCKDIM;
    BLOCKDIM = result_size / THREADDIM;

    gpu::pmeasure_many_target <<< (unsigned)(BLOCKDIM == 0 ? 1 : BLOCKDIM), (unsigned)THREADDIM >>> (
        psigpu.real,
        psigpu.imag,
        result_gpu,
        qnum_mask,
        result_size,
        1ull << (psigpu.qnum));


    cudaStatus = cudaMemcpy(&(mResult[0]), result_gpu, result_size * sizeof(STATE_T), cudaMemcpyDeviceToHost);

    cudaFree(result_gpu);

    return true;
}

bool GATEGPU::getState(QState &psi, QState &psigpu, size_t qnum)
{
    cudaError_t cuda_status;

    if (qnum < 30)
    {
        QSIZE Dim = 1ull << qnum;
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

void GATEGPU::gpuFree(double* memory)
{
    if (memory != nullptr)
    {
        cudaFree(memory);
    }
}
