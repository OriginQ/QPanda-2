#include "Core/QuantumNoise/NoiseModelV2.h"

#include <algorithm>
#include <vector>

#include "Core/Debugger/OriginDebug.h"
#include "Core/Debugger/Debug.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/Traversal.h"

USING_QPANDA

/*
  measure, rest and readout is NOT GateType, we create temporary enum label for handy coding
  max value of GateType is GateType::BARRIER_GATE = 36, now
*/
#define GATE_TYPE_MEASURE static_cast<GateType>(100)
#define GATE_TYPE_RESET static_cast<GateType>(101)
#define GATE_TYPE_READOUT static_cast<GateType>(102)

namespace
{
  Qnum QVec_to_Qnum(const QVec qvec)
  {
    Qnum qnum;
    for (auto qbit : qvec)
    {
      qnum.push_back(qbit->get_phy_addr());
    }
    return qnum;
  }

  QVec pick_qubit_by_addr(const QVec qvec, const Qnum qnum)
  {
    QVec find_qevc;
    for (auto qaddr : qnum)
    {
      QVec::const_iterator match_iter = std::find_if(qvec.begin(), qvec.end(),
                                                     [qaddr](Qubit *qbit)
                                                     {
                                                       if (qbit->get_phy_addr() == qaddr)
                                                       {
                                                         return true;
                                                       }
                                                       else
                                                       {
                                                         return false;
                                                       }
                                                     });
      if (match_iter != qvec.end())
        find_qevc.push_back(*match_iter);
    }
    return find_qevc;
  }

  void normlize(QStat &matrix, double p)
  {
    for (auto &val : matrix)
    {
      val *= p;
    }
  }

  /* qubit tensor state insert pos code from CPUImplQPU.h */
  inline int64_t _insert(int64_t value, size_t n)
  {
    int64_t number = 1ll << n;
    if (value < number)
    {
      return value;
    }

    int64_t mask = number - 1;
    int64_t x = mask & value;
    int64_t y = ~mask & value;
    return ((y << 1) | x);
  }

  inline int64_t _insert(int64_t value, size_t n1, size_t n2)
  {
    if (n1 > n2)
    {
      std::swap(n1, n2);
    }
    int64_t mask1 = (1ll << n1) - 1;
    int64_t mask2 = (1ll << (n2 - 1)) - 1;
    int64_t z = value & mask1;
    int64_t y = ~mask1 & value & mask2;
    int64_t x = ~mask2 & value;

    return ((x << 2) | (y << 1) | z);
  }
} // namespace

//--------------------------------------------------------------------------------------------------------------
/* same code from NoiseCPUImplQPU to make QuantamError */
void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type, double prob)
{
  add_noise_model(model, type, prob, std::vector<QVec>());
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model,
                                 const std::vector<GateType> &types,
                                 double prob)
{
  for (auto &type : types)
  {
    add_noise_model(model, type, prob, std::vector<QVec>());
  }
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<QVec> &qubits)
{
  size_t type_qubit_num = 0;
  if ((type >= GateType::P0_GATE && type <= U4_GATE) || GateType::I_GATE == type || GATE_TYPE_MEASURE == type || GATE_TYPE_RESET == type)
  {
    type_qubit_num = 1;
  }
  else if (type >= CU_GATE && type <= P11_GATE)
  {
    type_qubit_num = 2;
  }
  else
  {
    throw std::runtime_error("Error: noise qubit");
  }

  QuantumError quantum_error;
  quantum_error.set_noise(model, prob, type_qubit_num);

  std::vector<std::vector<size_t>> noise_qubits(qubits.size());
  for (size_t i = 0; i < qubits.size(); i++)
  {
    std::vector<size_t> addrs(qubits[i].size());
    for (size_t j = 0; j < qubits[i].size(); j++)
    {
      addrs[j] = qubits[i].at(j)->get_phy_addr();
    }
    noise_qubits[i] = addrs;
  }

  m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type,
                                 double prob, const QVec &qubits)
{
  std::vector<QVec> noise_qubits;
  noise_qubits.reserve(qubits.size());
  for (auto &val : qubits)
  {
    noise_qubits.push_back({val});
  }
  add_noise_model(model, type, prob, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model,
                                 const std::vector<GateType> &types, double prob,
                                 const QVec &qubits)
{
  std::vector<QVec> noise_qubits;
  noise_qubits.reserve(qubits.size());
  for (auto &val : qubits)
  {
    noise_qubits.push_back({val});
  }

  for (auto &type : types)
  {
    add_noise_model(model, type, prob, noise_qubits);
  }
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type,
                                 double T1, double T2, double t_gate)
{
  add_noise_model(model, type, T1, T2, t_gate, std::vector<QVec>());
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model,
                                 const std::vector<GateType> &types, double T1,
                                 double T2, double t_gate)
{
  for (auto &type : types)
  {
    add_noise_model(model, type, T1, T2, t_gate, std::vector<QVec>());
  }

  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type,
                                 double T1, double T2, double t_gate,
                                 const QVec &qubits)
{
  std::vector<QVec> noise_qubits;
  noise_qubits.reserve(qubits.size());
  for (auto &val : qubits)
  {
    noise_qubits.push_back({val});
  }

  add_noise_model(model, type, T1, T2, t_gate, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_noise_model(const NOISE_MODEL &model,
                                 const std::vector<GateType> &types, double T1,
                                 double T2, double t_gate, const QVec &qubits)
{
  std::vector<QVec> noise_qubits;
  noise_qubits.reserve(qubits.size());
  for (auto &val : qubits)
  {
    noise_qubits.push_back({val});
  }

  for (auto &type : types)
  {
    add_noise_model(model, type, T1, T2, t_gate, noise_qubits);
  }

  m_enable = true;
}

/* DECOHERENCE_KRAUS_OPERATOR */
void NoiseModel::add_noise_model(const NOISE_MODEL &model, const GateType &type,
                                 double T1, double T2, double t_gate,
                                 const std::vector<QVec> &qubits)
{
  size_t type_qubit_num = 0;
  if ((type >= GateType::P0_GATE && type <= U4_GATE) ||
      GateType::I_GATE == type || GATE_TYPE_MEASURE == type ||
      GATE_TYPE_RESET == type)
  {
    type_qubit_num = 1;
  }
  else if (type >= CU_GATE && type <= P11_GATE)
  {
    type_qubit_num = 2;
  }
  else
  {
    throw std::runtime_error("Error: noise qubit");
  }

  QuantumError quantum_error;
  quantum_error.set_noise(model, T1, T2, t_gate, type_qubit_num);

  std::vector<std::vector<size_t>> noise_qubits(qubits.size());
  for (size_t i = 0; i < qubits.size(); i++)
  {
    std::vector<size_t> addrs(qubits[i].size());
    for (size_t j = 0; j < qubits[i].size(); j++)
    {
      addrs[j] = qubits[i].at(j)->get_phy_addr();
    }
    noise_qubits[i] = addrs;
  }

  m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_measure_error(const NOISE_MODEL &model, double prob, const QVec &qubits)
{
  add_noise_model(model, GATE_TYPE_MEASURE, prob, qubits);
  m_enable = true;
}

void NoiseModel::add_measure_error(const NOISE_MODEL &model, double T1, double T2, double t_gate, const QVec &qubits)
{
  add_noise_model(model, GATE_TYPE_MEASURE, T1, T2, t_gate, qubits);
  m_enable = true;
}

void NoiseModel::add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices, const std::vector<double> &probs)
{
  add_mixed_unitary_error(type, unitary_matrices, probs, std::vector<QVec>());
  m_enable = true;
}

void NoiseModel::add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices, const std::vector<double> &probs, const QVec &qubits)
{
  std::vector<QVec> noise_qubits;
  noise_qubits.reserve(qubits.size());
  for (auto &val : qubits)
  {
    noise_qubits.push_back({val});
  }

  add_mixed_unitary_error(type, unitary_matrices, probs, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices, const std::vector<double> &probs, const std::vector<QVec> &qubits)
{
  size_t type_qubit_num = 0;
  if ((type >= GateType::P0_GATE && type <= U4_GATE) || GateType::I_GATE == type || GATE_TYPE_MEASURE == type || GATE_TYPE_RESET == type)
  {
    type_qubit_num = 1;
  }
  else if (type >= CU_GATE && type <= P11_GATE)
  {
    type_qubit_num = 2;
  }
  else
  {
    throw std::runtime_error("Error: noise qubit");
  }

  QuantumError quantum_error;
  quantum_error.set_noise(MIXED_UNITARY_OPRATOR, unitary_matrices, probs, type_qubit_num);

  std::vector<std::vector<size_t>> noise_qubits(qubits.size());
  for (size_t i = 0; i < qubits.size(); i++)
  {
    std::vector<size_t> addrs(qubits[i].size());
    for (size_t j = 0; j < qubits[i].size(); j++)
    {
      addrs[j] = qubits[i].at(j)->get_phy_addr();
    }
    noise_qubits[i] = addrs;
  }

  m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_reset_error(double p0, double p1, const QVec &qubits)
{
  QuantumError quantum_error;
  quantum_error.set_reset_error(p0, p1);

  std::vector<std::vector<size_t>> noise_qubits(qubits.size());
  for (size_t i = 0; i < qubits.size(); i++)
  {
    auto addr = qubits.at(i)->get_phy_addr();
    noise_qubits[i] = {addr};
  }

  m_quantum_noise.add_quamtum_error(GATE_TYPE_RESET, quantum_error, noise_qubits);
  m_enable = true;
}

void NoiseModel::add_readout_error(const std::vector<std::vector<double>> &probs_list, const QVec &qubits)
{
  QPANDA_ASSERT(0 == qubits.size() && 2 != probs_list.size(), "Error: readout paramters.");
  if (0 == qubits.size())
  {
    QuantumError quantum_error;
    quantum_error.set_readout_error(probs_list, 1);
    m_quantum_noise.add_quamtum_error(GATE_TYPE_READOUT, quantum_error, std::vector<std::vector<size_t>>());
  }
  else
  {
    for (size_t i = 0; i < qubits.size(); i++)
    {
      auto addr = qubits.at(i)->get_phy_addr();
      QuantumError quantum_error;
      auto iter = probs_list.begin() + 2 * i;
      quantum_error.set_readout_error({iter, iter + 2}, 1);
      m_quantum_noise.add_quamtum_error(GATE_TYPE_READOUT, quantum_error, {{addr}});
    }
  }
  m_readout_error_enable = true;
}

void NoiseModel::set_rotation_error(double error)
{
  m_rotation_angle_error = error;
}

double NoiseModel::rotation_error() const
{
  return m_rotation_angle_error;
}

bool NoiseModel::enabled() const
{
  return m_enable;
}

bool NoiseModel::readout_error_enabled() const
{
  return m_readout_error_enable;
}

const NoisyQuantum &NoiseModel::quantum_noise() const
{
  return m_quantum_noise;
}

//--------------------------------------------------------------------------------------------------------------
RandomEngine19937 NoiseGateGenerator::m_rng;

void NoiseGateGenerator::append_noise_gate(GateType gate_type, QVec target, NoisyQuantum &noise, AbstractNodeManager &noise_qc)
{
  Qnum qnum = QVec_to_Qnum(target);

  NoiseOp ops;
  Qnum effect_qubits;
  NOISE_MODEL noise_type;
  /* get gate noise ops, insert to new qprog as noise node */
  auto is_noise = noise.sample_noisy_op(gate_type, qnum, noise_type, ops, effect_qubits, m_rng);

  if (is_noise)
  {
    switch (noise_type)
    {
    /* for these casese, kraus operator is unitary */
    case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR:   // X
    case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR:   // Y
    case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR: // Z
    case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
    case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
      QPANDA_ASSERT(ops.size() != effect_qubits.size(), "Error: noise kruas");
      for (size_t i = 0; i < ops.size(); i++)
      {
        auto noise_target = pick_qubit_by_addr(target, {effect_qubits[i]});
        std::shared_ptr<OriginNoise> noise = std::make_shared<OriginNoise>(noise_target, ops[i]);
        noise_qc.pushBackNode(std::dynamic_pointer_cast<QNode>(noise));
      }
      break;

    /* for these casese, kraus operator is not unitary, we need chose operator base on qubit state dynamically */
    case NOISE_MODEL::DAMPING_KRAUS_OPERATOR:
    case NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR:
    {
      /* insert debug node before dynamic noise node to get current qubit state */
      noise_qc.pushBackNode(g_origin_debug);

      QVec noise_target;
      if (1 == target.size())
      {
        noise_target = pick_qubit_by_addr(target, {effect_qubits[0]});
      }
      else if (2 == target.size())
      {
        noise_target = pick_qubit_by_addr(target, {effect_qubits[0], effect_qubits[1]});
      }
      else
      {
        throw std::runtime_error(
            "Error: noise model canot support above two bits");
      }

      /* DynamicOriginNoise will pick out kraus operator on runtime base on current qubit state */
      auto noise = std::make_shared<DynamicOriginNoise<KrausOpGenerator>>(noise_target, KrausOpGenerator(effect_qubits, ops));
      noise_qc.pushBackNode(std::dynamic_pointer_cast<QNode>(noise));

      break;
    }

    case NOISE_MODEL::MIXED_UNITARY_OPRATOR:
    {
      if (1 == target.size())
      {
        std::shared_ptr<OriginNoise> noise = std::make_shared<OriginNoise>(target, ops[0]);
        noise_qc.pushBackNode(std::dynamic_pointer_cast<QNode>(noise));
      }
      else if (2 == target.size())
      {
        QPANDA_ASSERT(4 != ops[0].size(),
                      "mixed noise ops of two bits shoulde be dimesion 4");
        /* effect_qubits always in targets'id */
        auto noise_target = pick_qubit_by_addr(target, {effect_qubits[0], effect_qubits[1]});
        std::shared_ptr<OriginNoise> noise = std::make_shared<OriginNoise>(noise_target, ops[0]);
        noise_qc.pushBackNode(std::dynamic_pointer_cast<QNode>(noise));
      }
      else
      {
        throw std::runtime_error("Error: mixed noise only support two bits");
      }
      break;
    }

    default:
      throw std::runtime_error("Error: noise model not supported");
    }
  }
}

QStat NoiseGateGenerator::KrausOpGenerator::generate_op()
{
  QStat standard_matrix;
  auto random_p = m_rng.random_double();
  double sum_p = 0;
  double p = 0;
  bool is_get_noise_ops = false;

  /*
    similar code from NoiseCPUImplQPU
    try each noise op to get it's expectaion probability
    then normlize the op matrix and return to QVM for processing qubit state
  */
  for (size_t i = 0; i < m_noise_ops.size() - 1; i++)
  {
    p = kraus_expectation(m_qns, m_noise_ops[i]);
    sum_p += p;
    if (sum_p > random_p)
    {
      is_get_noise_ops = true;
      standard_matrix = m_noise_ops[i];
      QPANDA_ASSERT(std::abs(p) < FLT_EPSILON, "Error: normlize prob");
      normlize(standard_matrix, 1 / sqrt(p));
      break;
    }
  }

  if (!is_get_noise_ops)
  {
    auto res_p = 1 - sum_p;
    QPANDA_ASSERT(std::abs(res_p) < FLT_EPSILON, "Error: normlize prob");
    standard_matrix = m_noise_ops.back();
    normlize(standard_matrix, 1 / sqrt(res_p));
  }

  return standard_matrix;
}

double NoiseGateGenerator::KrausOpGenerator::kraus_expectation(const Qnum &qns,
                                                               const QStat &op)
{
  /* multiply kraus op with statvcter to get expected probability */
  double p = 0;
  qcomplex_t alpha, beta, gamma, phi;
  const auto &qstate_container = QPUDebugger::instance().get_qstate();
  std::vector<std::complex<double>> *double_qstate = qstate_container.double_state;
  std::vector<std::complex<float>> *float_qstate = qstate_container.float_state;

  int64_t qubit_num = 0;
  if (double_qstate)
  {
    qubit_num = std::log2(double_qstate->size());
  }
  else if (float_qstate)
  {
    qubit_num = std::log2(float_qstate->size());
  }
  else
  {
    throw std::runtime_error("no valid qstate to calculate kraus noise");
  }

  if (1 == qns.size())
  {
    size_t qn = qns.front();

    int64_t size = 1ll << (qubit_num - 1);
    int64_t offset = 1ll << qn;

    for (int64_t i = 0; i < size; i++)
    {
      int64_t real00_idx = _insert(i, qn);
      int64_t real01_idx = real00_idx | offset;

      if (double_qstate)
      {
        alpha = op[0] * (*double_qstate)[real00_idx] + op[1] * (*double_qstate)[real01_idx];
        beta = op[2] * (*double_qstate)[real00_idx] + op[3] * (*double_qstate)[real01_idx];
      }
      else if (float_qstate)
      {
        alpha = op[0] * (qcomplex_t)(*float_qstate)[real00_idx] + op[1] * (qcomplex_t)(*float_qstate)[real01_idx];
        beta = op[2] * (qcomplex_t)(*float_qstate)[real00_idx] + op[3] * (qcomplex_t)(*float_qstate)[real01_idx];
      }

      p += std::norm(alpha) + std::norm(beta);
    }
  }
  else if (2 == qns.size())
  {
    size_t qn_0 = qns[0];
    size_t qn_1 = qns[1];
    int64_t size = 1ll << (qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (qn_0 > qn_1)
    {
      std::swap(qn_0, qn_1);
    }

    for (int64_t i = 0; i < size; i++)
    {
      int64_t real00_idx = _insert(i, qn_0, qn_1);
      qcomplex_t phi00, phi01, phi10, phi11;
      if (double_qstate)
      {
        phi00 = (*double_qstate)[real00_idx];
        phi01 = (*double_qstate)[real00_idx | offset0];
        phi10 = (*double_qstate)[real00_idx | offset1];
        phi11 = (*double_qstate)[real00_idx | offset0 | offset1];
      }
      else if (float_qstate)
      {
        phi00 = (qcomplex_t)(*float_qstate)[real00_idx];
        phi01 = (qcomplex_t)(*float_qstate)[real00_idx | offset0];
        phi10 = (qcomplex_t)(*float_qstate)[real00_idx | offset1];
        phi11 = (qcomplex_t)(*float_qstate)[real00_idx | offset0 | offset1];
      }

      alpha = op[0] * phi00 + op[1] * phi01 + op[2] * phi10 + op[3] * phi11;
      beta = op[4] * phi00 + op[5] * phi01 + op[6] * phi10 + op[7] * phi11;
      gamma = op[8] * phi00 + op[9] * phi01 + op[10] * phi10 + op[11] * phi11;
      phi = op[12] * phi00 + op[13] * phi01 + op[14] * phi10 + op[15] * phi11;
      p += std::norm(alpha) + std::norm(beta) + std::norm(gamma) + std::norm(phi);
    }
  }
  else
  {
    throw std::runtime_error("Error: noise ops qubit");
  }

  return p;
}

RandomEngine19937 NoiseResetGenerator::m_rng;
void NoiseResetGenerator::append_noise_reset(GateType gate_type, QVec target, NoisyQuantum &noise, AbstractNodeManager &noise_qc)
{
  Qnum qnum = QVec_to_Qnum(target);

  NoiseOp ops;
  Qnum effect_qubits;
  /* get gate noise ops, insert to new qprog as noise gate */
  auto is_noise = noise.sample_noisy_op(GATE_TYPE_RESET, qnum, ops, effect_qubits, m_rng);

  if (is_noise)
  {
    QPANDA_ASSERT(2 != ops.size(), "Reset error ops error");
    std::shared_ptr<OriginNoise> noise = std::make_shared<OriginNoise>(target, ops[0]);
    noise_qc.pushBackNode(std::dynamic_pointer_cast<QNode>(noise));
  }
}

RandomEngine19937 NoiseReadOutGenerator::m_rng;
void NoiseReadOutGenerator::append_noise_readout(const NoiseModel &noise_model, std::map<std::string, bool> &result)
{
  for (auto &it : result)
  {
    /* trans cbit string to qubit id */
    size_t qn = atoi(it.first.c_str() + 1);

    std::vector<std::vector<double>> readout_error;
    auto is_noise = const_cast<NoisyQuantum &>(noise_model.quantum_noise()).sample_noisy_op(qn, readout_error, m_rng);
    if (is_noise)
    {
      if (it.second)
      {
        it.second = m_rng.random_discrete(readout_error[1]);
      }
      else
      {
        it.second = m_rng.random_discrete(readout_error[0]);
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------------------

void NoiseProgGenerator::execute(std::shared_ptr<AbstractQGateNode> cur_node,
                                 std::shared_ptr<QNode> parent_node)
{
  QNodeDeepCopy::execute(cur_node, parent_node);

  QuantumGate *qgate = cur_node->getQGate();
  GateType gate_type = (GateType)qgate->getGateType();
  QVec target;
  cur_node->getQuBitVector(target);

  /* bypass controlled gate for noise_model not support */
  if (0 == cur_node->getControlQubitNum())
  {
    /* parent_node is new node copied from cur_node */
    NoiseGateGenerator::append_noise_gate(gate_type, target, m_qnoise, *std::dynamic_pointer_cast<AbstractNodeManager>(parent_node));
  }
}

void NoiseProgGenerator::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node)
{
  /* add virtual measure noise node before real measure node */
  Qubit *target = cur_node->getQuBit();
  NoiseGateGenerator::append_noise_gate(GATE_TYPE_MEASURE, {target}, m_qnoise, *std::dynamic_pointer_cast<AbstractNodeManager>(parent_node));

  QNodeDeepCopy::execute(cur_node, parent_node);
}

void NoiseProgGenerator::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
{
  QNodeDeepCopy::execute(cur_node, parent_node);

  Qubit *target = cur_node->getQuBit();
  NoiseResetGenerator::append_noise_reset(GATE_TYPE_RESET, {target}, m_qnoise, *std::dynamic_pointer_cast<AbstractNodeManager>(parent_node));
}