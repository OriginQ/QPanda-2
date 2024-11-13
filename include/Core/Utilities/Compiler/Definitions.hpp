#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qc 
{
class QFRException : public std::invalid_argument {
  std::string msg;

public:
  explicit QFRException(std::string m)
      : std::invalid_argument("QFR Exception"), msg(std::move(m)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }
};

using QBit = std::uint32_t;
using Bit = std::uint64_t;

template <class IdxType, class SizeType>
using Register = std::pair<IdxType, SizeType>;
using QuantumRegister = Register<QBit, std::size_t>;
using ClassicalRegister = Register<Bit, std::size_t>;
template <class RegisterType>
using RegisterMap = std::map<std::string, RegisterType, std::greater<>>;
using QuantumRegisterMap = RegisterMap<QuantumRegister>;
using ClassicalRegisterMap = RegisterMap<ClassicalRegister>;
using RegisterNames = std::vector<std::pair<std::string, std::string>>;

using Targets = std::vector<QBit>;

using BitString = std::bitset<4096>;

// floating-point type used throughout the library
using fp = double;

constexpr fp PARAMETER_TOLERANCE = 1e-13;

 static constexpr fp qcPI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);

static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);
static constexpr fp TAU = static_cast<fp>(
    6.283185307179586476925286766559005768394338798750211641950L);
static constexpr fp E = static_cast<fp>(
    2.718281828459045235360287471352662497757247093699959574967L);

static constexpr size_t OUTPUT_INDENT_SIZE = 2;

// forward declaration
class Operation;

// supported file formats
enum class Format : uint8_t {
  Real,
  OpenQASM2,
  OpenQASM3,
  GRCS,
  TFC,
  QC,
  Tensor
};

using DAG = std::vector<std::deque<std::unique_ptr<Operation>*>>;
using DAGIterator = std::deque<std::unique_ptr<Operation>*>::iterator;
using DAGReverseIterator =
    std::deque<std::unique_ptr<Operation>*>::reverse_iterator;
using DAGIterators = std::vector<DAGIterator>;
using DAGReverseIterators = std::vector<DAGReverseIterator>;

/**
 * @brief 64bit mixing hash (from MurmurHash3)
 * @details Hash function for 64bit integers adapted from MurmurHash3
 * @param k the number to hash
 * @returns the hash value
 * @see https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */
[[nodiscard]] constexpr std::size_t murmur64(std::size_t k) noexcept {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

/**
 * @brief Combine two 64bit hashes into one 64bit hash
 * @details Combines two 64bit hashes into one 64bit hash based on
 * boost::hash_combine (https://www.boost.org/LICENSE_1_0.txt)
 * @param lhs The first hash
 * @param rhs The second hash
 * @returns The combined hash
 */
[[nodiscard]] constexpr std::size_t
combineHash(const std::size_t lhs, const std::size_t rhs) noexcept {
  return lhs ^ (rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2));
}

/**
 * @brief Extend a 64bit hash with a 64bit integer
 * @param hash The hash to extend
 * @param with The integer to extend the hash with
 * @return The combined hash
 */
constexpr void hashCombine(std::size_t& hash, const std::size_t with) noexcept {
  hash = combineHash(hash, with);
}

/**
 * @brief Function used to mark unreachable code
 * @details Uses compiler specific extensions if possible. Even if no extension
 * is used, undefined behavior is still raised by an empty function body and the
 * noreturn attribute.
 */
[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
}

	static std::map<std::string, std::function<double(double, double)>> _binary_operation =
	{
		{"+", [](double  lval,double rval) {return lval + rval; }},
		{"-", [](double lval,double rval) {return lval - rval; } },
		{"*", [](double lval,double rval) {return lval * rval; } },
		{"/", [](double lval,double  rval) {return lval / rval; } },
	};

	/**
	* @brief  Saves the expression containing the variable
	* @ingroup Utilities
	*/
	class Exp
	{
	public:
		struct Content
		{
			std::string var_name;
			std::string op_specifier;
			double const_value;
		};

		enum ContentType
		{
			VAR_NAME = 0,
			OP_EXPR,
			CONST_VAL,
		};

		Exp(std::string name)
		{
			m_content.var_name = name;
			m_content_type = VAR_NAME;
		}

		Exp(std::shared_ptr<Exp> left_exp_ptr, std::shared_ptr<Exp> right_exp_ptr, std::string op)
		{
			m_left_exp_ptr = left_exp_ptr;
			m_right_exp_ptr = right_exp_ptr;
			m_content.op_specifier = op;
			m_content_type = OP_EXPR;
		}

		Exp(double val)
		{
			m_content.const_value = val;
			m_content_type = CONST_VAL;
		}

		~Exp() {}

		/**
		* @brief   clone Exp class
		* @return  std::shared_ptr<Exp>   Exp  class shared ptr
		*/
		std::shared_ptr<Exp> clone() { return std::make_shared<Exp>(*this); }

		void set_formal_actual_var_map(std::map <std::string, double> name_val_map)
		{
			m_formal_actual_var_map = name_val_map;
			if (m_content_type == OP_EXPR)
			{
				m_left_exp_ptr->set_formal_actual_var_map(name_val_map);
				m_right_exp_ptr->set_formal_actual_var_map(name_val_map);
			}
		}

		/**
		* @brief   evaluation
		* @return  double   operation rusult
		*/
		double eval()
		{
			if (m_content_type == VAR_NAME)
			{
				std::string var_name = m_content.var_name;
				auto iter_actual = m_formal_actual_var_map.find(var_name);
				if (iter_actual == m_formal_actual_var_map.end())
				{
					//QCERR("get actual val error!");
					throw std::runtime_error("get actual val error!");
				}
				return iter_actual->second;
			}
			else if (m_content_type == OP_EXPR)
			{
				double left_val = m_left_exp_ptr->eval();
				double right_val = m_right_exp_ptr->eval();

				auto iter_func = _binary_operation.find(m_content.op_specifier);
				if (iter_func == _binary_operation.end())
				{
					//QCERR("get binary operation  function error!");
					throw std::runtime_error("get binary operation  function error!");
				}
				return iter_func->second(left_val, right_val);
			}
			else if (m_content_type == CONST_VAL)
			{
				return m_content.const_value;
			}
			else
			{
				//QCERR("content typer error!");
				throw std::invalid_argument("content typer error!");
			}
		}

	private:
		std::shared_ptr<Exp> m_left_exp_ptr;
		std::shared_ptr<Exp> m_right_exp_ptr;
		int m_content_type;
		Content m_content;
		std::map<std::string, double> m_formal_actual_var_map;
	};

	struct RegParamInfo
	{
		std::string reg_name;
		int reg_index;
	};

	struct GateOperationInfo
	{
		std::string op_id;
		std::vector<RegParamInfo> regs_vec;
		std::vector<std::shared_ptr<Exp>> angles_vec;
	};

	struct GataFuncInfo
	{
		std::string func_name;
		std::vector<std::string> angle_names_vec;
		std::vector<std::string> reg_names_vec;
		std::vector<GateOperationInfo> ops_vec;
	};
}