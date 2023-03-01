#ifndef QPANDA_UTILS_MACRO_H
#define QPANDA_UTILS_MACRO_H

//#define PRINT_DEBUG_MESSAGE

#ifdef PRINT_DEBUG_MESSAGE
#define PRINT_CUDA_DEBUG_MESSAGE printf("file:%s  line:%ld\n", __FILE__, __LINE__);
#define PRINT_DEBUG_MESSAGE std::cout << "file:" << __FILE__ << "  line:" << __LINE__ << "  function:" << __FUNCTION__ << std::endl;
#else
#define PRINT_CUDA_DEBUG_MESSAGE
#define PRINT_DEBUG_MESSAGE
#endif

#define PRINT_DEBUG_CUDAINFO                                                                   \
    std::cout << "*************************************************************" << std::endl; \
    system("nvidia-smi");                                                                      \
    std::cout << "*************************************************************" << std::endl;

#define CHECK_CUDA(CUDA_FUNC)                                          \
    {                                                                  \
        PRINT_DEBUG_MESSAGE                                            \
        cudaError_t CUDA_ERROR = CUDA_FUNC;                            \
        if ((int)CUDA_ERROR != (int)(CUDA_SUCCESS))                    \
        {                                                              \
            throw(std::runtime_error(cudaGetErrorString(CUDA_ERROR))); \
        }                                                              \
    }
#endif