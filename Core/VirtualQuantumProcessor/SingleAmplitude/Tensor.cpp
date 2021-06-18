#include "Tensor.h"
#include "QPandaConfig.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif
USING_QPANDA

static size_t getNum(long long src, size_t *mask_array, size_t size,size_t der_size)
{
    size_t der = 0;
    for (size_t i = 0; i < der_size; i++)
    {
        bool value = (src >> (size - mask_array[i] - 1)) & 1;
        der += value << (der_size - i -1);
    }

    return der;
}
static int get_num_threads(size_t rank)
{
    int nthreds = 1;
    if (rank > 9)
    {
#ifdef USE_OPENMP
        nthreds = omp_get_max_threads();
#endif
    }
    return nthreds;
}


CPUComplexTensor::~CPUComplexTensor()
{
    if (nullptr != m_tensor)
    {
        free(m_tensor);
        m_tensor = nullptr;
    }
}

size_t CPUComplexTensor::getRank() const
{
    return m_rank;
}

qcomplex_data_t CPUComplexTensor::getElem(size_t num)
{
    return m_tensor[num];
}

void CPUComplexTensor::mulElem(size_t num, qcomplex_data_t elem)
{
    if (num > 1ull << m_rank)
    {
        QCERR("mulElem error");
        throw std::runtime_error("mulElem error");
    }

    m_tensor[num] *= elem;
}

size_t CPUComplexTensor::getMaxRank() const
{
    return m_max_rank;
}

void CPUComplexTensor::dimIncrement(size_t increment_size)
{
    if(m_rank + increment_size > m_max_rank)
    {
        QCERR("dimIncrement error");
        throw std::runtime_error("dimIncrement error");
    }

    auto size = 1ull << m_rank;
    m_rank = m_rank + increment_size;
    auto new_size = 1ull << m_rank;
    auto new_tensor = (qcomplex_data_t *)calloc(new_size, sizeof(qcomplex_data_t));

    if (nullptr == new_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }

    int threads = get_num_threads(m_rank);
    int j;
#pragma omp parallel for num_threads(threads) private(j)
    for (long long i = 0; i < size; i++)
    {
        for (j = 0; j < 1 << increment_size; j++)
            {
                new_tensor[(i << increment_size) + j] = m_tensor[i];
            }
     }

    free(m_tensor);
    m_tensor = new_tensor;
}

void CPUComplexTensor::getSubTensor(size_t num, int value)
{
    if (num > m_rank)
    {
        QCERR("getSubTensor error");
        throw std::runtime_error("getSubTensor error");
    }

    auto size = 1ull << m_rank;
    auto sub = m_rank - num;
    qsize_t step = 1ull << sub;

    m_rank = this->m_rank - 1;
    auto new_size = 1ull << m_rank;
    auto new_tensor = (qcomplex_data_t *)calloc(new_size, sizeof(qcomplex_data_t));
    if (nullptr == new_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }

    int threads = get_num_threads(m_rank);
    size_t j;
    size_t k = 0;
#pragma omp parallel for num_threads(threads) private(j,k)
    for (long long i = 0; i < size; i += step * 2)
    {
        k = i / (2 * step);
        for (j = i; j < i + step; j++)
        {
            if (0 == value)
            {
                new_tensor[j - k * step] = m_tensor[j];
            }
            else if (1 == value)
            {
                new_tensor[j - k * step] = m_tensor[j + step];
            }
            else
            {
                throw std::runtime_error("error");
            }
        }
    }

    free(m_tensor);
    m_tensor = new_tensor;
}

void CPUComplexTensor::dimDecrement(size_t num)
{
    if ((num > m_rank) || (m_rank == 0))
    {
        QCERR("dimDecrement error");
        throw std::runtime_error("dimDecrement error");
    }

    auto size = 1ull << m_rank;
    auto sub = m_rank - num;
    qsize_t step = 1ull << sub;
    m_rank = this->m_rank - 1;

    auto new_size = 1ull << m_rank;
    auto new_tensor = (qcomplex_data_t *)calloc(new_size, sizeof(qcomplex_data_t));
    if (nullptr == new_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }

    size_t k = 0;
    size_t step_num = size / step;
    int threads = get_num_threads(m_rank);

    if (step_num <= 4)
    {
        for (long long i = 0; i < size; i += step * 2)
        {
            k = i / (2 * step);
#pragma omp parallel for num_threads(threads)
            for (long long j = i; j < i + step; j++)
            {
                new_tensor[j - k * step] = (m_tensor[j] + m_tensor[j + step]);
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(threads) private(k)
        for (long long i = 0; i < size; i += step * 2)
        {
            k = i / (2 * step);
            for (long long j = i; j < i + step; j++)
            {
                new_tensor[j - k * step] = (m_tensor[j] + m_tensor[j + step]);
            }
        }
    }

    free(m_tensor);
    m_tensor = new_tensor;
}


qcomplex_data_t *CPUComplexTensor::getTensor()
{
    return m_tensor;
}

void CPUComplexTensor::mul(ComplexTensor &other, size_t *mask_array)
{
    int64_t size = 1ll << m_rank;
    int threads = get_num_threads(m_rank);
#pragma omp parallel for num_threads(threads)
    for (int64_t i = 0; i < size; i++)
    {
        auto num = getNum(i, mask_array, getRank(), other.getRank());
        mulElem(i, other.getElem(num));
    }
}

ComputeBackend CPUComplexTensor::getBackend()
{
    return m_backend;
}


CPUComplexTensor::CPUComplexTensor(const CPUComplexTensor &old)
    :m_max_rank(old.m_max_rank), m_rank(old.m_rank)
{
    auto size = 1ull << old.m_rank;
    m_tensor = (qcomplex_data_t *)calloc(size, sizeof(qcomplex_data_t));

    if (nullptr == m_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }
    int threads = get_num_threads(m_rank);
#pragma omp parallel for num_threads(threads)
    for (long long i = 0; i < size; i++)
    {
            m_tensor[i] = old.m_tensor[i];
    }
}

CPUComplexTensor::CPUComplexTensor(size_t rank, qstate_t &tensor, size_t max_rank)
    :m_max_rank(max_rank), m_rank(rank), m_backend(ComputeBackend::CPU)
{
    auto size = 1ull << rank;
    m_tensor = (qcomplex_data_t *)calloc(size, sizeof(qcomplex_data_t));

    if (nullptr == m_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }
    int threads = get_num_threads(m_rank);
#pragma omp parallel for num_threads(threads)
    for (long long i = 0; i < size; i++)
    {
            m_tensor[i] = tensor[i];
    }
}

CPUComplexTensor::CPUComplexTensor(size_t rank, qcomplex_data_t *tensor, size_t max_rank)
    :m_max_rank(max_rank), m_rank(rank), m_backend(ComputeBackend::CPU)
{
    auto size = 1ull << rank;
    m_tensor = (qcomplex_data_t *)calloc(size, sizeof(qcomplex_data_t));

    if (nullptr == m_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }
    int threads = get_num_threads(m_rank);
#pragma omp parallel for num_threads(threads)
    for (long long i = 0; i < size; i++)
    {
            m_tensor[i] = tensor[i];
    }
}

CPUComplexTensor &CPUComplexTensor::operator =(const CPUComplexTensor &old)
{
    if (this == &old)
    {
        return *this;
    }

    m_rank = old.m_rank;
    m_max_rank = old.m_max_rank;
    auto size = 1ull << old.m_rank;
    m_backend = old.m_backend;
    auto new_tensor = (qcomplex_data_t *)calloc(size, sizeof(qcomplex_data_t));

    if (nullptr == new_tensor)
    {
        QCERR("calloc_fail");
        throw calloc_fail();
    }
    int threads = get_num_threads(m_rank);
#pragma omp parallel for num_threads(threads)
    for (long long i = 0; i < size; i++)
    {
        new_tensor[i] = old.m_tensor[i];
    }

    free(m_tensor);
    m_tensor = new_tensor;
    return *this;
}

size_t ComplexTensor::getRank() const
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }
    return m_tensor->getRank();
}

qcomplex_data_t ComplexTensor::getElem(size_t num)
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }
    return m_tensor->getElem(num);
}

void ComplexTensor::dimIncrement(size_t num)
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    m_tensor->dimIncrement(num);
}

void ComplexTensor::getSubTensor(size_t num, int value)
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    m_tensor->getSubTensor(num, value);
}

void ComplexTensor::dimDecrement(size_t num)
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    m_tensor->dimDecrement(num);
}

qcomplex_data_t *ComplexTensor::getTensor()
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    return m_tensor->getTensor();
}

void ComplexTensor::mul(ComplexTensor &other, size_t *mask_array)
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    m_tensor->mul(other, mask_array);
}

ComplexTensor::ComplexTensor(std::shared_ptr<AbstractComplexTensor> tensor)
{
    m_tensor = tensor;
}

size_t ComplexTensor::getMaxRank() const
{
    if (!m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }
    return m_tensor->getMaxRank();
}

ComputeBackend ComplexTensor::getBackend()
{
    return m_tensor->getBackend();
}

ComplexTensor &ComplexTensor::operator=(const ComplexTensor &old)
{
    if ((!m_tensor) || (!old.m_tensor))
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }

    m_tensor = old.m_tensor;
    return *this;
}

ComplexTensor::ComplexTensor(const ComplexTensor &old)
{
    if (!old.m_tensor)
    {
        QCERR("m_tensor is null");
        throw std::runtime_error("m_tensor is null");
    }
    auto backend = old.m_tensor->getBackend();
    std::shared_ptr<AbstractComplexTensor> share_temp;
    AbstractComplexTensor *p_tensor = old.m_tensor.get();

    switch (backend)
    {
    case ComputeBackend::CPU:
    {
        CPUComplexTensor *cpu_tensor = dynamic_cast<CPUComplexTensor *>(p_tensor);
        share_temp.reset(new CPUComplexTensor(*cpu_tensor));
        break;
    }
    default:
        throw std::runtime_error("backend error");
        break;
    }

    m_tensor = share_temp;
}

ComplexTensor::ComplexTensor(ComputeBackend backend, size_t rank, qstate_t &tensor, size_t max_rank)
{
    switch (backend)
    {
    case ComputeBackend::CPU:
        m_tensor = std::make_shared<CPUComplexTensor>(rank, tensor, max_rank);
        break;
    default:
        throw std::runtime_error("backend error");
        break;
    }
}

ComplexTensor::ComplexTensor(ComputeBackend backend, size_t rank, qcomplex_data_t *tensor, size_t max_rank)
{
    switch (backend)
    {
    case ComputeBackend::CPU:
        m_tensor = std::make_shared<CPUComplexTensor>(rank, tensor, max_rank);
        break;
    default:
        throw std::runtime_error("backend error");
        break;
    }
}

ComplexTensor::~ComplexTensor()
{
    if (m_tensor)
    {
        m_tensor.reset();
    }
}
