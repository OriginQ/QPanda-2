#include "AbstractOptimizer.h"

namespace QPanda
{

    AbstractOptimizer::AbstractOptimizer() :
        m_optimized_para(2, 0),
        m_disp(false),
        m_adaptive(false),
        m_xatol(1e-4),
        m_fatol(1e-4),
        m_test_value(0.0),
        m_max_fcalls(0),
        m_max_iter(0),
        m_restore_from_cache_file(false)
    {
        m_result.message = "No exec.";
    }

    AbstractOptimizer::~AbstractOptimizer()
    {

    }

}
