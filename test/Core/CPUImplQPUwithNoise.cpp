#include "QPanda.h"
#include "gtest/gtest.h"
#include <cmath>

USING_QPANDA

int shot = 5000;
int bit_len = 3;
double allowed_error = 0.03;

// compare tools for test
template <typename map_t>
bool compare_result(const map_t &r1, const map_t &r2, double total, double allowed_error)
{
  if (r1.size() != r2.size())
  {
    std::cout << "***ERROR:*** r1.size " << r1.size() << " != r2.size " << r2.size() << std::endl;
    return false;
  }

  for (auto &pair : r1)
  {
    if (std::abs((double)pair.second - (double)r2.at(pair.first)) / total > allowed_error)
    {
      std::cout << pair.first << "***ERROR:*** |r1 " << pair.second << " - r2 " << r2.at(pair.first) << "|/" << total << " > " << allowed_error << std::endl;
      return false;
    }
  }
  return true;
}

template <typename map_t>
void print_result(const map_t &r1, const map_t &r2)
{
  std::cout << "r1 size " << r1.size() << " | r2 size " << r2.size() << std::endl;

  for (auto &pair : r1)
  {
    std::cout << pair.first << " r1 " << pair.second << " | r2 " << r2.at(pair.first) << std::endl;
  }
}

// compare tools end

TEST(CPUImplQPUwithNoise, BITFLIP_KRAUS_OPERATOR_compare)
{
  auto qpool = OriginQubitPool::get_instance();
  auto cmem = OriginCMem::get_instance();

  auto qvec = qpool->qAllocMany(bit_len);
  auto cvec = cmem->cAllocMany(bit_len);

  auto prog1 = createEmptyQProg();
  auto prog2 = createEmptyQProg();
  auto prog3 = createEmptyQProg();

  prog1 << X(qvec[0]) << X(qvec[1]) << MeasureAll(qvec, cvec);
  prog2 << X(qvec[0]) << SWAP(qvec[1], qvec[0]) << MeasureAll(qvec, cvec);
  prog3 << H(qvec[0]) << CNOT(qvec[0], qvec[1]) << X(qvec[1]) << MeasureAll(qvec, cvec);

  auto noise_qvm = NoiseQVM();
  noise_qvm.init();
  noise_qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.3);
  noise_qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);

  auto cpu_qvm = CPUQVM();
  cpu_qvm.init();
  NoiseModel noise;
  noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.3);
  noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);

  auto r1 = noise_qvm.runWithConfiguration(prog1, cvec, shot);
  auto r2 = cpu_qvm.runWithConfiguration(prog1, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog2, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog2, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog3, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog3, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  noise_qvm.finalize();
  cpu_qvm.finalize();

  qpool->qFreeAll();
  cmem->cFreeAll();
}

TEST(CPUImplQPUwithNoise, BIT_PHASE_FLIP_KRAUS_OPERATOR_compare)
{
  auto qpool = OriginQubitPool::get_instance();
  auto cmem = OriginCMem::get_instance();

  auto qvec = qpool->qAllocMany(bit_len);
  auto cvec = cmem->cAllocMany(bit_len);

  auto prog1 = createEmptyQProg();
  auto prog2 = createEmptyQProg();
  auto prog3 = createEmptyQProg();

  prog1 << X(qvec[0]) << P(qvec[0], PI) << MeasureAll(qvec, cvec);
  prog2 << X(qvec[0]) << SWAP(qvec[1], qvec[0]) << MeasureAll(qvec, cvec);
  prog3 << X(qvec[0]) << S(qvec[0]) << MeasureAll(qvec, cvec);

  auto noise_qvm = NoiseQVM();
  noise_qvm.init();
  noise_qvm.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::P_GATE, 0.3);
  noise_qvm.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::S_GATE, 0.5);

  auto cpu_qvm = CPUQVM();
  cpu_qvm.init();
  NoiseModel noise;
  noise.add_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::P_GATE, 0.3);
  noise.add_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::S_GATE, 0.5);

  auto r1 = noise_qvm.runWithConfiguration(prog1, cvec, shot);
  auto r2 = cpu_qvm.runWithConfiguration(prog1, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog2, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog2, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog3, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog3, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  noise_qvm.finalize();
  cpu_qvm.finalize();

  qpool->qFreeAll();
  cmem->cFreeAll();
}

TEST(CPUImplQPUwithNoise, DAMPING_KRAUS_OPERATOR_compare)
{
  auto qpool = OriginQubitPool::get_instance();
  auto cmem = OriginCMem::get_instance();

  auto qvec = qpool->qAllocMany(bit_len);
  auto cvec = cmem->cAllocMany(bit_len);

  auto prog1 = createEmptyQProg();
  auto prog2 = createEmptyQProg();
  auto prog3 = createEmptyQProg();

  prog1 << H(qvec[1]) << CNOT(qvec[1], qvec[0]) << X(qvec[1]) << MeasureAll(qvec, cvec);
  prog2 << H(qvec[0]) << SWAP(qvec[1], qvec[0]) << MeasureAll(qvec, cvec);
  prog3 << H(qvec[0]) << X(qvec[0]) << MeasureAll(qvec, cvec);

  auto noise_qvm = NoiseQVM();
  noise_qvm.init();
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.3);
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::SWAP_GATE, 0.7);

  auto cpu_qvm = CPUQVM();
  cpu_qvm.init();
  NoiseModel noise;
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.3);
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::SWAP_GATE, 0.7);

  auto r1 = noise_qvm.runWithConfiguration(prog1, cvec, shot);
  auto r2 = cpu_qvm.runWithConfiguration(prog1, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog2, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog2, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  r1 = noise_qvm.runWithConfiguration(prog3, cvec, shot);
  r2 = cpu_qvm.runWithConfiguration(prog3, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  noise_qvm.finalize();
  cpu_qvm.finalize();

  qpool->qFreeAll();
  cmem->cFreeAll();
}

inline QCircuit build_U_fun(QVec qubits)
{
  QCircuit cir_u;
  cir_u << RY(qubits[0], PI);
  return cir_u;
}

TEST(CPUImplQPUwithNoise, PHASE_DAMPING_OPRATOR)
{
  auto qpool = OriginQubitPool::get_instance();
  auto cmem = OriginCMem::get_instance();

  auto cqv = qpool->qAllocMany(bit_len);
  auto tqv = qpool->qAllocMany(1);
  auto cv = cmem->cAllocMany(bit_len);

  /*
    use qpe to estimate phase
    to show phase damping noise how effect

    RY gate eigen state is 1/sqrt(2)*(|0>-i|1>), with input angle is PI, qpe result should be 010 in binary, which is 2 in decimal
    so theta = 2/(2^bit_len) = 1/4

    if noise added to pauli Z gate
    target bit is not eigen vector of RY gate
  */
  auto prog1 = QProg();
  prog1 << H(tqv[0]) << S(tqv[0]) << Z(tqv[0]) << build_QPE_circuit(cqv, tqv, build_U_fun) << MeasureAll(cqv, cv);

  auto noise_qvm = NoiseQVM();
  noise_qvm.init();
  /*
    for PHASE_DAMPING_OPRATOR, is't real effect prob = (1-sqrt(1-p))/2,
    we use 1 here, so real effect prob is 0.5, means 50% of Z gate will be phase damping
  */
  noise_qvm.set_noise_model(NOISE_MODEL::PHASE_DAMPING_OPRATOR, GateType::PAULI_Z_GATE, 1);

  auto cpu_qvm = CPUQVM();
  cpu_qvm.init();
  NoiseModel noise;
  noise.add_noise_model(NOISE_MODEL::PHASE_DAMPING_OPRATOR, GateType::PAULI_Z_GATE, 1);

  auto r1 = noise_qvm.runWithConfiguration(prog1, cv, shot);
  auto r2 = cpu_qvm.runWithConfiguration(prog1, cv, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, shot, allowed_error));

  noise_qvm.finalize();
  cpu_qvm.finalize();

  qpool->qFreeAll();
  cmem->cFreeAll();
}

TEST(CPUImplQPUwithNoise, MIX_ALL_NOISE)
{

  auto qpool = OriginQubitPool::get_instance();
  auto cmem = OriginCMem::get_instance();

  auto qvec = qpool->qAllocMany(bit_len);
  auto cvec = cmem->cAllocMany(bit_len);

  QProg prog = createEmptyQProg();

  prog << X(qvec[1]) << SWAP(qvec[2], qvec[1]) << CNOT(qvec[2], qvec[0]) << MeasureAll(qvec, cvec);

  std::vector<std::vector<double>> prob_lists{{0.9, 0.1}, {0.1, 0.9}};

  auto noise_qvm = NoiseQVM();
  noise_qvm.init();
  noise_qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.7);
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::SWAP_GATE, 0.3);
  noise_qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.5);
  noise_qvm.set_readout_error(prob_lists, qvec[2]);
  noise_qvm.set_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.5, qvec[2]);
  noise_qvm.set_reset_error(0.5, 0.5, qvec[1]);

  auto cpu_qvm = CPUQVM();
  cpu_qvm.init();
  NoiseModel noise;
  noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.7);
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::SWAP_GATE, 0.3);
  noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.5);
  noise.add_readout_error(prob_lists, qvec[2]);
  noise.add_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.5, qvec[2]);
  noise.add_reset_error(0.5, 0.5, qvec[1]);

  auto r1 = noise_qvm.runWithConfiguration(prog, cvec, shot);
  auto r2 = cpu_qvm.runWithConfiguration(prog, cvec, shot, noise);
  // print_result(r1, r2);
  ASSERT_TRUE(compare_result(r1, r2, allowed_error, shot));

  noise_qvm.finalize();
  cpu_qvm.finalize();

  qpool->qFreeAll();
  cmem->cFreeAll();
}