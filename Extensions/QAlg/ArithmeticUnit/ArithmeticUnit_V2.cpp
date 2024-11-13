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

size_t auxBitSize(size_t t, ADDER type){
  auto adder = AdderFactory::getInstance().createAdder(type);
  return adder->auxBitSize(t);
}

QCircuit QAdd_V2(QVec &a, QVec &b, QVec &aux, ADDER type /* = ADDER::CDKM_RIPPLE*/)
{
  if ((a.size() == 0) || (a.size() != b.size()))
  {
    QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
  }

  auto adder = AdderFactory::getInstance().createAdder(type);

  /* use two's complement for negative addition */
  /*
    overflow rule is(in complement representation):
      1.if a, b is negative, sum is positive, overflow
      2.if a, b is positive, sum is negative, overflow
      3.if a, b have different sign, won't overflow
      note sum only take fixed bit size same as a or b, did not take account carry bit
    
    overflow = a.b.s' ⊕ a'.b'.s  (short for sign)
    
    we save a.b and a'.b' to ancil bit, but from the truth table, we found it's hard to restore ancil bits
      b  a.b  a'.b'  sum  o  a 
      1   1     0     1   0  1  *
      1   1     0     0   1  1
      0   0     0     0   0  1
      0   0     0     1   0  1  *
      1   0     0     0   0  0  +
      1   0     0     1   0  0
      0   0     1     0   0  0  +
      0   0     1     1   1  0
    as sum is saved in b after addition, so we can only use a, s, o to restore ancil a.b and a'.b'
    but found two undistinguishable pairs marked with '*' and '+', it is unreversible goal, can not restore ancil
    this may overkill

    so we used another rule:
      overflow = unsign_carry_out ⊕ full_carry_out
  
    this way can only implemented inside adder for get unsigned num carry
    but this will miss one situation: sum is negative 0
    take 3 bit complement binary for example:
      -3   101
    + -1   111
    -----------
      -4  1100
    cut fix size result(3 bits), it'is 100, which is negative 0
    unsign_carry = 1
    full_carry = 1
    so overflow = 0, missed
    but negative 0 will give other trace, see QComplement_V2
  */
  QPANDA_ASSERT(aux.size()==0, "aux at least have size 1");
  QCircuit qc;
  qc << QComplement_V2(a, aux.front())
     << QComplement_V2(b, aux.front())
     << adder->QAdd(b, a, aux, ADDER_MODE::OF)
     << QComplement_V2(a, aux.front())
     << QComplement_V2(b, aux.front());
  return qc;
}

QCircuit QSub_V2(QVec &a, QVec &b, QVec &aux, ADDER type /* = ADDER::CDKM_RIPPLE*/)
{
  QCircuit qc;
  qc << X(b.back())
     << QAdd_V2(a, b, aux, type)
     << X(b.back());
  return qc;
}

/**
 * @note circuits is reversible, so it's one to one
 * input positive 0 or negative 0, output are both postivie 0, so ancil is different for reversibility 
 */
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

QCircuit QComplement_V2(QVec &a, Qubit *aux)
{
  int a_n = a.size();
  QVec unsign_a(a.begin(), a.end() - 1);
  Qubit *a_sign = a.back();

  /* 
      see [T. G. Draper, Addition on a Quantum Computer, 2000] part 4, how to compute approximate qft threshold
      log2(n) is good enough
    */
  int aqft_threshold = (std::max)((int)std::floor(std::log2(a_n)), 6);

  QCircuit qc = CreateEmptyCircuit();
  /* if negative, flip unsign party bits */
  for (auto qbit : unsign_a)
  {
    qc << CNOT(a_sign, qbit);
  }
  /* if negateiv, use aux bit for temporary |1> for addition */
  qc << CNOT(a_sign, aux);

  /* DrapperQftAdder */
  qc << DraperQFTAdder::AQFT(a, aqft_threshold);

  for (int target_id = a_n - 1; target_id >= 0; target_id--)
  {
    /* add |1> to target bit */
    if (target_id < aqft_threshold)
    {
      qc << CR(aux, a[target_id], 2 * PI / (1 << (target_id + 1)));
    }
  }

  qc << DraperQFTAdder::inverseAQFT(a, aqft_threshold);
  /* if negative, restore temporary bit */
  qc << CNOT(a_sign, aux);
  return qc;
}
QPANDA_END