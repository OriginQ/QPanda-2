#include "QAlg/ArithmeticUnit/ArithmeticUnit_V2.h"
#include "QAlg/ArithmeticUnit/ArithmeticUnit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/ArithmeticUnit/Adder.h"
#include "Core.h"

QPANDA_BEGIN

class AdderFactory
{
public:
  ~AdderFactory() = default;

  static AdderFactory &getInstance()
  {
    static AdderFactory ins;
    return ins;
  }

  std::shared_ptr<AbstractAdder> createAdder(ADDER type)
  {
    switch (type)
    {
    case ADDER::CDKM_RIPPLE:
      return std::make_shared<CDKMRippleAdder>();
    case ADDER::DRAPER_QFT:
      return std::make_shared<DraperQFTAdder>();
    case ADDER::VBE_RIPPLE:
      return std::make_shared<VBERippleAdder>();
    case ADDER::DRAPER_QCLA:
      return std::make_shared<DraperQCLAAdder>();
    default:
      QCERR_AND_THROW(std::runtime_error, "selected UNKNOWN adder");
    }
  }

private:
  AdderFactory() = default;
};

QCircuit QAdd_V2(QVec &a, QVec &b, QVec &aux, ADDER type /* = ADDER::CDKM_RIPPLE*/)
{
  if ((a.size() == 0) || (a.size() != b.size()))
  {
    QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
  }

  auto adder = AdderFactory::getInstance().createAdder(type);

  QVec unsign_a(a.begin(), a.end() - 1);
  QVec unsign_b(b.begin(), b.end() - 1);
  Qubit *a_sign = a.back();
  Qubit *b_sign = b.back();
  QVec overflow(aux.back());
  overflow += aux;
  /* use two's complement for negative addition */
  QCircuit qc;
  qc << QComplement_V2(a, aux)
     << QComplement_V2(b, aux)
     << adder->QAdd(b, a, aux, ADDER_MODE::OF)
     << QComplement_V2(a, aux)
     << QComplement_V2(b, aux);
  return qc;
}

QCircuit QSub_V2(QVec &a, QVec &b, QVec &aux, ADDER type /* = ADDER::CDKM_RIPPLE*/)
{
  auto len = b.size();
  QCircuit qc;
  qc << X(b[len - 1])
     << QAdd_V2(a, b, aux, type);
  // if (ADDER::CDKM_RIPPLE == type)
  // {
  qc << X(b[len - 1]);
  // }
  return qc;
}

QCircuit QComplement_V2(QVec &a, QVec &aux)
{
  /* 
    we proposed circute for two's complement based on QCLA adder
    filp a then add one bit b_0 which is |1>, if a is negative, otherwise keep not changed
  */
  /* aux should be |0...0>, not checked here */
  /* highest bit of a is sign */
  QVec unsign_a(a.begin(), a.end() - 1);
  Qubit *sign = a.back();

  int n = a.size();

  /* sizeof(z) = n, sizeof(x) = n - log n, total auxiliary size at least  2 * n + 1 - log2(n) */
  std::stringstream error_msg("auxiliary bits size should at leat 2 * n + 1 - std::floor(std::log2(n)) = ");
  error_msg << int(2 * n - std::floor(std::log2(n)));
  QPANDA_ASSERT(aux.size() < int(2 * n - std::floor(std::log2(n))), error_msg.str());

  /*
    z[i+1] = g[i, i + 1]
    a[i] = p[i, i + 1] for i > 0
    x for ancillary space to save p[i,j], j>i+1
  */
  QVec z(aux.begin(), aux.begin() + n);
  QVec x(aux.begin() + n, aux.begin() + int(2 * n - std::floor(std::log2(n))));

  /* resue z[0] as b_0 */
  Qubit *b_0 = z[0];

  DraperQCLAAdder::Propagate p(a, x);

  QCircuit qc;
  /*
    step 0: 
    1. set b_0 |1>, if sign negative
    2. flip bits of unsign_a, if sign negative
  */
  qc << X(b_0).control(sign);
  for (int i = 0; i < unsign_a.size(); i++)
  {
    qc << X(unsign_a[i]).control(sign);
  }

  /*
    QCLA algorithm
    step 1:
    g[0,1] = (a[0] * b_0) ⊕ g[0,1]
  */
  if (1 < n)
    qc << X(z[1]).control({a[0], b_0});

  /* 
    step 2:
    add b_0 to a_0
    this sets a[i] = p[i, i+1]
  */
  qc << CNOT(b_0, a[0]);

  auto qc_carry = createEmptyCircuit();
  /*
    step 3: P rounds
   
    P_t[m] = p[pow(2,t)*m, pow(2,t)*(m+1)]
    G[m] = g[x,m]     x can be any value
   
    for 1 <= t <= floor(log(n)) - 1:
      for 1 <= m < floor(n/pow(2,t))):
        P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
  */
  for (int t = 1; t <= std::floor(std::log2(n)) - 1; t++)
  {
    for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
    {
      /* skip carry out in p[n-1, x] */
      if (n - 1 == pow(2, t) * m ||
          n - 1 == pow(2, t - 1) * 2 * m ||
          n - 1 == pow(2, t - 1) * (2 * m + 1))
      {
        break;
      }
      auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
      auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
      auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
      qc_carry << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
    }
  }

  /*
    step 3: G rounds
    for  1 <= t <= floor(log n):
      for 0 <= m < floor(n/pow(2,t)):
        G[pow(2,t)*m + pow(2,t)] ⊕= G[pow(2,t)*m + pow(2,t−1)]P_t−1[2m + 1]
  */
  for (int t = 1; t <= std::floor(std::log2(n)); t++)
  {
    for (int m = 0; m < std::floor(n / std::pow(2.0, t)); m++)
    {
      /* skip carry out in z[n] = g[x, n] */
      if (n - 1 == pow(2, t) * (m + 1) - 1 ||
          n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
          n - 1 == pow(2, t - 1) * (2 * m + 1))
      {
        break;
      }
      auto g_target = z[pow(2, t) * (m + 1)];
      auto g_control = z[pow(2, t - 1) * (2 * m + 1)];
      auto p_t_1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
      qc_carry << Toffoli(g_control, p_t_1, g_target);
    }
  }

  /*
    step 3: C rounds
    for floor(log (2n/3)) >= t >= 1:
      for 1 <= m <= floor((n-pow(2,t-1))/pow(2,t)):
        G[pow(2,t)*m + pow(2,t−1)] ⊕= G[pow(2,t)*m]P_t−1[2m]
  */
  for (int t = floor(log2(2.0 * n / 3.0)); t >= 1; t--)
  {
    for (int m = 1; m <= std::floor((n - std::pow(2.0, t - 1)) / pow(2.0, t)); m++)
    {
      if (n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
          n - 1 == pow(2, t) * m - 1 ||
          n - 1 == pow(2, t - 1) * (2 * m))
      {
        break;
      }
      auto g_target = z.at(pow(2, t - 1) * (2 * m + 1));
      auto g_control = z.at(pow(2, t) * m);
      auto p_t_1 = p(pow(2, t - 1) * (2 * m), pow(2, t - 1) * (2 * m + 1));
      qc_carry << Toffoli(g_control, p_t_1, g_target);
    }
  }

  /*
    step 3: reverse P rounds
    for floor(log n) >= t >= 1:
      for 1 <= m < floor(n/pow(2,t)):
        P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
  */
  for (int t = floor(log2(n)); t >= 1; t--)
  {
    for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
    {
      if (n - 1 == pow(2, t) * m ||
          n - 1 == pow(2, t - 1) * 2 * m ||
          n - 1 == pow(2, t - 1) * (2 * m + 1))
      {
        break;
      }
      auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
      auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
      auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
      qc_carry << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
    }
  }
  qc << qc_carry;
  /* after step 3 z[i]=c_i */

  /*
    step 4:
    a[i] = a[i] ⊕ z[i] for i in [1,n)
    now a[i] = sum[i]
  */
  for (int i = 1; i < n; i++)
  {
    qc << CNOT(z.at(i), a.at(i));
  }

  /*
    step 5:
    flip a[i] for i in [0,n-1)
   */
  for (int i = 0; i < n - 1; i++)
  {
    qc << X(a[i]);
  }

  /* 
    step 6:
    a[i] = a[i] ⊕ b[i] for i in [1,n-1)
    is skiped as b_0 is only one bit
  */

  /* step 7: revers step 3 */
  qc << qc_carry.dagger();

  /*
    step 8:
    for i in [1,n-1):
      a[i] = a[i] ⊕ b[i] 
    skiped as b[i] only have one bit b[0] = b_0
  */

  /* 
    step 9:
    z[i+1] = z[i+1] ⊕(a[i]*b[i]) for i in [0,n-1)
    is
    z[1] = z[1] ⊕(a[0]*b_0)
    skiped if n is 1 to eliminate highest carry
  */
  if (1 < n)
  {
    qc << X(z.at(1)).control({a[0], b_0});
  }

  /* 
   step 11:
   flip b_0 if sign negative
  */
  qc << X(b_0).control(sign);

  /*
    step 12:
    flip a[i] for i in [0, n-1)
  */
  for (size_t i = 0; i < n - 1; i++)
  {
    qc << X(a[i]);
  }

  return qc;
}
QPANDA_END