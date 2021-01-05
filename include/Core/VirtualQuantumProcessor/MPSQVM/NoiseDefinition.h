#ifndef NOISE_DEFINITION_H
#define NOISE_DEFINITION_H

#include <vector>
#include <stdio.h>
#include <iostream>
#include <Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h>

QPANDA_BEGIN

std::vector<QStat>  get_noise_model_karus_matrices(NOISE_MODEL model, const std::vector<double>& params);

std::vector<double> get_noise_model_unitary_probs(NOISE_MODEL model, double param);

std::vector<QStat>  get_noise_model_unitary_matrices(NOISE_MODEL model, double param);

QPANDA_END


#endif  //!NOISE_DEFINITION_H 