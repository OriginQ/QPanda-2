#pragma once

#include "QPandaConfig.h"


#include <vector>
#include <complex>

namespace PilotQVM {
    typedef std::complex<double> Complex_;
    using QStat = std::vector<Complex_>;

    /**< Chip current status */
    enum class ChipState
    {
        ONLINE = 0,                /**< Chip Online */
        OFFLINE = 1,            /** Chip offline */
    };

    /**< Chip back-end computing mode */
    enum class CalcMethod
    {
        CHIP_CALC = 0,         /**< Chip computing, only allowed when the chip is online */
        SIMULATE_CALC = 1,        /**< Simulation calculation */
    };

    enum class MeasureType
    {
        UNKNOW_MEASURE_TYPE = 0,
        MONTE_CARLO_MEASURE = 1,
        PMEASURE = 2,
        EXPECTATION /**< 求期望 */
    };

    /**< Calculation backend type */
    using ExecuteBackendType = uint32_t;

#define ERR_BACKEND   0X1FFFFFF           /**< Wrong chip type */
#define ANY_QUANTUM_CHIP   0X2000000      /**< For chip tasks for which the user does not specify a target computing backend, Sinan system customizes the target computing backend */
#define ANY_CLUSTER_BACKEND   0X2000001   /**< The cluster computing backend is customized by the system */

    /** Types of quantum computers (simulation computing, real chip Computing) */
    enum class QMType: int
    {
        FULL_AMPLITUDE = 0,            /**<  Full amplitude simulation */
        NOISE,                        /**<  Noise simulation */
        PARTIAL_AMPLITUDE,            /**<  Partial amplitude simulation */
        SINGLE_AMPLITUDE,            /**<  Single amplitude simulation */
        MPS,                         /**<  MPs simulation */
        REAL_CHIP                   /**<  Real chip computing */
    };

    enum class TaskStatus
    {
        UNDEFINED_STATE = 0,         /**< Unknown status, generally refers to no relevant information is queried */
        SENT = 1,                    /**< The system receives a calculation task */
        RUNNING = 2,                 /**< The target calculation task is being executed */
        FINISHED = 3,                /**< The target calculation task has been completed and the table calculation is successful */
        FAILED = 4,                  /**< Target calculation task calculation failed */
        CONVERTED = 5,               /**< The target computing task has completed compilation and conversion, and has not been executed yet */

        CONVERTING = 9,              /**< The target computing task is in compilation conversion */

        PARALLEL_CONVERTING = 22,    /**< The target computing task is undergoing parallel compilation conversion */

        CANCELLED = 35               /**< The target calculation task has been canceled */
    };
    using PilotTaskStatus = PilotQVM::TaskStatus; /**< Used to distinguish taskstatus definitions in qpanda */

    enum class TaskType : int
    {
        UNDEFIND_TASK_TYPE = -1,        /**< Unknown task type */
        MEASURE = 0,                    /**< Monte Carlo measurement task */
        PMEASURE = 1,                   /**< Pmeasure measurement task */
        QST_TASK = 2,                  /**< QST task */
        QST_DENSITY = 3,                /**< QST density matrix calculation task */
        FIDELITY = 4,                   /**< Fidelity calculation task */
        REAL_EXPECTATION = 5,           /**<  Real chip for expectation! */
        MEASURE_UNOPTIMIZATE,           /**< No optimized measurement task */
        MEASURE_UNMAPPING,               /**< No mapping measurement task */

    };

    enum class ErrorCode : uint32_t {
        NO_ERROR_FOUND = 0,                      /**< No error */
        DATABASE_ERROR,                          /**< Database error */
        ORIGINIR_ERROR,                          /**< Originir syntax error */
        JSON_FIELD_ERROR,                        /**< Json error */
        BACKEND_CALC_ERROR,                      /**< Back end calculation error */
        ERR_TASK_BUF_OVERFLOW,                   /**< System task queue full */
        EXCEED_MAX_QUBIT,                        /**< Qubit exceeds */
        ERR_UNSUPPORT_BACKEND_TYPE,                /**< Unsupported backend type */
        EXCEED_MAX_CLOCK = 8,                      /**< Quantum program timing out */
        ERR_UNKNOW_TASK_TYPE,                    /**< Wrong task type */
        ERR_QVM_INIT_FAILED,                     /**< Virtual machine initialization failed */
        ERR_QCOMPILER_FAILED,                    /**< Compilation failed */
        ERR_PRE_ESTIMATE,                        /**< Task estimation error */
        ERR_MATE_GATE_CONFIG,                    /**< Metadata configuration error */
        ERR_FIDELITY_MATRIX,                     /**< Fidelity density matrix error */
        ERR_QST_PROG,                            /**< QST program error */
        ERR_EMPTY_PROG = 16,                    /**< Empty quantum program */
        ERR_QUBIT_SIZE,                          /**< The number of qubits is wrong and does not match the target logical gate */
        ERR_QUBIT_TOPO,                          /**< Qubit topology error */
        ERR_QUANTUM_CHIP_PROG,                   /**< Quantum chip computing task program error */
        ERR_REPEAT_MEASURE,                      /**< Duplicate measure */
        ERR_OPERATOR_DB,                         /**< Database operation error */
        ERR_TASK_STATUS_BUF_OVERFLOW,            /**< System task status information is full */
        ERR_BACKEND_CHIP_TASK_SOCKET_WRONG,        /**< Communication failed when the backend module sends tasks to the backend */
        CLUSTER_SIMULATE_CALC_ERR = 24,            /**< Cluster calculation error */
        ERR_SCHEDULE_CHIP_TOPOLOGY_SUPPORTED,    /**< Scheduling error, the chip topology does not support line operation */
        ERR_TASK_CONFIG,                         /**< Task configuration error */
        ERR_NOT_FOUND_APP_ID,                    /**< Unknown app ID */
        ERR_NOT_FOUND_TASK_ID,                   /**< Unknown task ID */
        ERR_PARSER_SUB_TASK_RESULT,              /**< Error parsing subtask result */
        ERR_SYS_CALL_TIME_OUT,                   /**< System call timeout */
        ERR_TASK_TERMINATED,                    /**< Task terminated */
        ERR_INVALID_URL = 32,                    /**< URL address error */
        ERR_PARAMETER,                           /**< Parameter error */
        ERR_QPROG_LENGTH,                        /**< Quantum circuit length error */
        ERR_CHIP_OFFLINE,                        /**< The chip is offline, and the real computing mode cannot be set */
        UNDEFINED_ERROR,                         /**< unknown error */
        ERR_SUB_GRAPH_OUT_OF_RANGE,                /**< The sub graph exceeds the maximum limit and cannot be calculated */
        ERR_TCP_INIT_FATLT,                        /*< The client failed to connect to the server through TCP >*/
        ERR_TCP_SERVER_HALT,                    /*< TCP server down >*/
        CLUSTER_BASE = 1000
    };
}
