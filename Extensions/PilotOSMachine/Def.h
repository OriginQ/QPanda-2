#ifndef QPANDA_PILOT_DEF_H
#define QPANDA_PILOT_DEF_H

#include "QPandaConfig.h"


#include <ctime>
#include <chrono>
#include "QPanda.h"
#include "Extensions/PilotOSMachine/PilotException.h"


// message macor define
/** OS Task json key string */
#define MSG_APP_ID                        "app_id"
#define MSG_TASK_ID                    "task_id"
#define MSG_IR                                "originir"
//#define MSG_IR_LEN                        "ir_len"    /**< length of OriginIR */
#define MSG_QM_TYPE                    "qm_type"           /**< quantum computer type */
#define MSG_BACKEND                    "backend"           /**< backend type */
#define MSG_BACKEND_PART            "backend_part"
#define MSG_TOPIC                    "topic"
#define MSG_IS_SINGLE_OR_CONNECTED    "qubit_kind"
#define MSG_PRIORITY                     "priority"
#define MSG_TASK_CONFIG            "config"
#define MSG_PRE_CONFIG            "pre_config"
#define MSG_START_TIME            "start_time"
#define MSG_LOGIC_QUBIT_TOPO      "logic_qubit_topo"     /**< topology information of logic qubit in circuit */

/** RealChip Cluster task preprocess json key */
#define MSG_QUBIT_NUM            "qubit_num"
#define MSG_TIME_SEQ                    "time_seq"
#define MSG_REQUIRED_NODE        "required_node"         /**< number of nodes used by simulation task */
#define MSG_REQUIRED_RAM          "required_ram"         /**< number of ram used by simulation task */
#define MSG_SUB_CIR_SIZE          "sub_cir_size"         /**< number of disassembled molecular graphs */

/** RealChip task config json key */
#define MSG_TASK_TYPE            "task_type"
#define MSG_SHOT                        "shot"
#define MSG_IS_OPTIMIZE           "is_optimize"
#define MSG_IS_MAPPING            "is_mapping"
#define MSG_IS_AMEND              "is_amend"            /**< Whether to correct the result */

#define MSG_TASK_RESULT                "task_result"          /**< task result */
#define MSG_TASK_QST_DENSITY        "qst_density"            /**< density matrix of task qst */
#define MSG_TASK_QST_FIDELITY        "qst_fidelity"                /**< fidelity of task qst */
#define MSG_TASK_CHIP_EXE_TIME        "chip_time"    

#define MSG_ERROR_CODE           "error_code"
#define MSG_ERROR_INFO           "error_info"

#define MSG_STATE_QUERY_RESULT   "state_query_result"   /**< Status information query results */
#define MSG_SINGLE_TASK_QUERY_RESULT   "single_task_query_result"   /**< Query results of single task status information */
#define MSG_SUB_APP_ID           "sub_app_id"          /**< App information corresponding to subtasks */
#define MSG_SUB_TASK_ID          "sub_task_id"          /**< ID information corresponding to subtasks */
#define MSG_SUB_TASK_IR            "sub_task_ir"     /**< Subtask IR */
#define MSG_INSTRUCTIONS         "instructions"         /**< Compiled output instructions */
#define MSG_TARGET_QUBIT         "target_qubit"         /**< Qubit information corresponding to the task */
#define MSG_SUB_TASK_QUBIT       "sub_task_qubit"       /**< Qubit information corresponding to subtasks */
#define MSG_TASK_INFO            "qprog_info"           /**< Quantum program information */
#define MSG_LOGIC_CONN_QBLOCK       "logic_qblock"       /**< Line logic bit connected block */
#define MSG_PHY_CONN_QBLOCK       "phy_qblock"       /**< Chip physical bit connectivity block */


/** OS internal message type */
#define MSG_TYPE                                            "msg_type"     /**< OS internal message type */
#define MSG_COMPILED_CHIP_TASK                "compiled_chip_task"       /**< Compile and send the compiled task information to the backend */
#define MSG_COMPILED_CLUSTER_TASK        "compiled_cluster_task"    /**< Compile and send the compiled task information to the backend */
#define MSG_SCHEDULED_CHIP_TASK            "scheduled_chip_task"       /**< Post scheduling task message */
#define MSG_SCHEDULED_CLUSTER_TASK    "scheduled_cluster_task"  /**< Post scheduling task message */
#define MSG_SCHEDULED_TASK_ERR            "scheduled_task_error"       /**< Scheduling task error */

#define MSG_BACKEND_REQ  "backend_req"        /**< Back end request message */
#define MSG_CLUSTER_RESP     "cluster_resp"            /**< Cluster response message */

#define MSG_CHIP_RES_SET  "chip_res_set"            /**< Set chip resources */
#define MSG_CHIP_RES_RECOVER  "chip_res_recover"            /**< Recover chip resources */
#define MSG_IDLE_QUBIT            "idle_qubit"                    /**< free qubit */
#define MSG_TOTAL_QUBITS        "total_qubits"                    /**< Total qubit quantity */
#define MSG_MAX_CLOCK_CYCLE        "max_clock_cycle"

#define MSG_CLUSTER_RES_SET                "cluster_res_set"            /**< Set cluster resources */
#define MSG_CLUSTER_RES_RECOVER    "cluster_res_recover" /**< Recover cluster resources */
#define MSG_TOTAL_NODES                "total_nodes"                    /**< Total number of nodes */
#define MSG_MEMORY_PER_NODE   "memory_per_node"        /**< Memory per node */

#define MSG_GET_TASK_BY_ID       "get_task_by_id"       /**< Get task by task ID */

#define MSG_DICT_KEY                "key"
#define MSG_DICT_VALUE          "value"
#define MSG_SERVER_STATE_CHECK   "server_state_check"          /**< System service status check message */
#define MSG_MERGED_MAPPING_QUBIT "merged_mapping_qubit"       /**< The qubit mapping relation of combined quantum circuits */


#define MSG_SET_CALC_METHOD_REQ            "set_calc_method_req"        /**< Set calculation mode */
#define MSG_SET_CALC_METHOD_REP        "set_calc_method_rep"   /**< Set calculation mode response message */
#define MSG_CHIP_STATE                "chip_state"            /**< Chip status */
#define MSG_CALC_METHOD            "calc_method"        /**< Chip computing mode*/
#define MSG_SET_RESULT                "set_result"            /**< Setting result: bool true succeeds, false fails */ 
#define MSG_TASK_CANCEL_REQ            "task_cancel_req"       /**< Cancel task request message */
#define MSG_TASK_CANCEL_RESP        "task_cancel_resp"      /**< Cancel task response message */

#define MSG_CHIP_STATE_NOTIFY        "chip_state_notify"   /**< Chip online status notification, back-end service to resource service */
#define MSG_CALC_METHOD_NOTIFY        "calc_method_notify"   /**< Chip computing mode notification, back-end service to resource service */

#define MSG_SET_TASK_PRIORITY                "set_task_priority"    /**< Set task priority */
#define MSG_TASK_PRIORITY_UPDATE_RESULT       "task_priority_update_result"    /**< Task priority update result */
#define MSG_IS_OK                             "is_ok"                          /**< Success? */
#define MSG_CMD_UUID                          "cmd_uuid"     /**< System command message ID, which is different from taskid */
#define MSG_BACKEND_STATE_QUERY   "backend_state_query"     /**< Get backend status (including chip and cluster backend) */
#define MSG_BACKEND_STATE_RESP    "backend_state_response"  /**< Back end status feedback command */

#define MSG_BACKEND_PROPS_PULL    "bprops_pull"    
#define MSG_BACKEND_PROPS_PUSH    "bprops_push"
#define MSG_PUSH_STATE  "push_state"
#define MSG_SCHDULE_QUEUE_QUERY  "schdule_queue_qry"    /**< Get the queuing information of the scheduling queue */
#define MSG_SCHDULE_QUEUE_RESP   "schdule_queue_resp"   /**< Scheduling queue queuing return information */
#define MSG_MAX_QUERY_TASK_SIZE  "max_query_task_size"  /**< Maximum number of query tasks */

#define MSG_QUEUE_TASK_SIZE     "queue_task_size"       /**< Number of queued tasks in the task queue */
#define MSG_QUEUE_TASK          "queue_task"            /**< Information of queued tasks in task queue */

#define MSG_BPROPS_INIT_END   "bprops_init_end"    

#define MSG_BPROPS_SET                "bprops_set"   
#define MSG_BACKEND_PROPS    "bprops"    

#define MSG_MEASURE_TYPE                    "measure_type"

/** DB Key String */
#define TASK_ID_STR ("TaskId")
#define PARENT_ID_STR ("ParentId")
#define TASK_STATE_STR ("TaskState")

#define RegisterPilotServer(pilot_server) \
int main(int argc, char* argv[]){\
    bool is_clean_shm = true;\
    if (argc == 2 && strcmp(argv[1], "-c") == 0) {\
        is_clean_shm = false;\
    }\
    static pilot_server _##pilot_server##_instance;\
    if (_##pilot_server##_instance.init(is_clean_shm)){\
        _##pilot_server##_instance.run();\
    }\
    return 0;}

#define AUTO_CB(cb_func_) AutoCallBack __a_cb_(cb_func_);

#if defined(_MSC_VER)
#define _get_pid() _getpid()

#elif defined(__linux__)
#define _get_pid() getpid()

#endif // _MSC_VER

#endif

// !DEF_H