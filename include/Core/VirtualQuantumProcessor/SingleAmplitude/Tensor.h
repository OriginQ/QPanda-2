#ifndef TENSOR_H__
#define TENSOR_H__

#include <complex>
#include <vector>
#include <memory>
#include <map>
#include "Core/Utilities/Tools/QPandaException.h"

typedef float qdata_t;
typedef size_t qsize_t;
typedef std::complex<qdata_t> qcomplex_data_t;
typedef std::vector<qcomplex_data_t> qstate_t;

enum class ComputeBackend
{
    CPU,
    /*GPU*/
};
class ComplexTensor;

class AbstractComplexTensor
{
public:
    virtual ~AbstractComplexTensor() {};
    virtual size_t getRank() const = 0;
    virtual size_t getMaxRank() const = 0;
    virtual qcomplex_data_t getElem(size_t num) = 0;
    virtual void dimIncrement(size_t) = 0;
    virtual void getSubTensor(size_t num, int value) = 0;
    virtual void dimDecrement(size_t num) = 0;
    virtual qcomplex_data_t *getTensor() = 0;
    virtual void mul(ComplexTensor & other, size_t * mask_array) = 0;
    virtual ComputeBackend getBackend() = 0;
};

class CPUComplexTensor : public AbstractComplexTensor
{
public:
    virtual ~CPUComplexTensor();
    virtual size_t getRank() const;
    virtual qcomplex_data_t getElem(size_t num);
    virtual size_t getMaxRank() const;

    virtual void dimIncrement(size_t increment_size) ;
    virtual void getSubTensor(size_t num,int value);
    virtual void dimDecrement(size_t num) ;
    virtual qcomplex_data_t *getTensor();

    virtual void mul(ComplexTensor & other, size_t * mask_array);
    virtual ComputeBackend getBackend();
	void mulElem(size_t num, qcomplex_data_t elem);

    CPUComplexTensor(const CPUComplexTensor& old);
    CPUComplexTensor(size_t rank, qstate_t & tensor,size_t max_rank);
    CPUComplexTensor(size_t rank, qcomplex_data_t * tensor,size_t max_rank);
    CPUComplexTensor& operator = (const CPUComplexTensor &old);

protected:
    size_t m_max_rank;
    size_t m_rank{0};
    qcomplex_data_t * m_tensor{nullptr};
    ComputeBackend m_backend{ComputeBackend::CPU};
};

class ComplexTensor : public AbstractComplexTensor
{
public:
    virtual size_t getRank() const ;
    virtual qcomplex_data_t getElem(size_t num);
    virtual void dimIncrement(size_t num);

    virtual void getSubTensor(size_t num, int value);
    virtual void dimDecrement(size_t num);
    virtual qcomplex_data_t *getTensor();
    virtual void mul(ComplexTensor & other, size_t * mask_array);

    virtual size_t getMaxRank() const;
    virtual ComputeBackend getBackend();

    ComplexTensor(std::shared_ptr<AbstractComplexTensor> tensor);
    ComplexTensor & operator=(const ComplexTensor &old);
    ComplexTensor(const ComplexTensor &old);

    ComplexTensor(ComputeBackend backend, size_t rank, qstate_t & tensor,size_t max_rank);
    ComplexTensor(ComputeBackend backend, size_t rank, qcomplex_data_t * tensor,size_t max_rank);
    virtual ~ComplexTensor();
private:
    std::shared_ptr<AbstractComplexTensor> m_tensor;
};


#endif // TENSOR_H_
