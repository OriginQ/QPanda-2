#pragma once

#include <string>
#include <vector>

#include "OSDef.h"

#define COMMON_MODE true
#define SPECIAL_MODE false

struct CalcConfig {
    uint32_t backend_id;
    uint32_t shot { 1000 };
    uint32_t task_type{ 0 };
    uint32_t pulse_period{ 0 };
    uint32_t point_lable{ 0 };
    uint32_t priority{ 0 };     /* task priority (0-10, default is 0) */
    bool is_amend {true};
    bool is_mapping {true};
    bool is_optimization {true};
    bool is_post_process{ true };
    bool is_prob_counts { false };
    std::string hamiltonian;
    std::string ir;
    std::vector<std::string>ir_vec;
    std::vector<uint32_t> specified_block;
    std::string task_describe;
};

struct NoiseConfig {
    bool mode{ COMMON_MODE };
    std::string script;
    std::string ir;
    std::string noise_learning_result_file;     /* filename */
    int shots;
    int samples;
    int loops;                                  /**< Number of random lines */
    double noise_strength;
    bool is_em_compute;                         /**< Whether to perform EM(Error mitigation) calculations */
    std::vector<uint32_t> circuit_depth_list;
};