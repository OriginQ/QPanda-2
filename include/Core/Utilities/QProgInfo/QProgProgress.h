#ifndef _QPROG_PROGRESS_H
#define _QPROG_PROGRESS_H

#include "Core/Utilities/Tools/ReadWriteLock.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <unordered_map>
#include <atomic>

QPANDA_BEGIN

/**
 * @brief for access processed QGate num while QVM running qprog
 *        use singleton for fewest affect other's interface
 * 
 * @note if qprog has not started or has finished, return 0
 * 
 */
class QProgProgress
{
private:
    QProgProgress() = default;
    ~QProgProgress() = default;
    std::unordered_map<uint64_t, size_t> m_prog_exec_gates;
    std::mutex m_mutex;

public:

    static QProgProgress &getInstance()
    {
        static QProgProgress obj;
        return obj;
    }

    /**
     * @brief start/end trace QProgExecution
     * 
     * @param exec_id we use QProgExecution obj address as uniq id
     */
    void prog_start(uint64_t exec_id);
    void prog_end(uint64_t exec_id);

    /**
     * @brief get the processed gate num object
     * 
     * @note if QProgExecution not be record, return 0
     */
    size_t get_processed_gate_num(uint64_t exec_id);

    /**
     * @brief update processed gate num
     * 
     * @param count default add 1 count
     * @return size_t if QProgExecution id is registed, return counted gates num,
     *         else return 0
     */
    size_t update_processed_gate_num(uint64_t exec_id, size_t count = 1);
};

QPANDA_END

#endif