#ifndef GENERIC_TEMPLATE_UINTWIDE_T_2018_10_02_H_
  #define GENERIC_TEMPLATE_UINTWIDE_T_2018_10_02_H_

  ///////////////////////////////////////////////////////////////////
  //  Copyright Christopher Kormanyos 1999 - 2018.                 //
  //  Distributed under the Boost Software License,                //
  //  Version 1.0. (See accompanying file LICENSE_1_0.txt          //
  //  or copy at http://www.boost.org/LICENSE_1_0.txt)             //
  ///////////////////////////////////////////////////////////////////

  #include <algorithm>
  #include <array>
  #include <cstddef>
  #include <cstdint>
  #include <iterator>
  #include <limits>
  #include <type_traits>

  #if defined(WIDE_INTEGER_DISABLE_IOSTREAM)
  #else
  #include <iomanip>
  #include <istream>
  #include <ostream>
  #include <sstream>
  #endif

  namespace wide_integer { namespace generic_template {

  // Forward declaration of uintwide_t.
  template<const std::size_t Digits2,
           typename LimbType = std::uint32_t>
  class uintwide_t;

  // Forward declarations of non-member binary add, sub, mul, div, mod of (uintwide_t op IntegralType).
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator+(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator-(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator*(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator/(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator%(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);

  // Forward declarations of non-member binary add, sub, mul, div, mod of (uintwide_t op IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator+(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator-(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator*(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator/(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator%(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  // Forward declarations of non-member binary add, sub, mul, div, mod of (IntegralType op uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator+(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator-(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator*(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator/(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator%(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  // Forward declarations of non-member binary logic operations of (uintwide_t op uintwide_t).
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator|(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator^(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator&(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);

  // Forward declarations of non-member binary logic operations of (uintwide_t op IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator|(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator^(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator&(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  // Forward declarations of non-member binary binary logic operations of (IntegralType op uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator|(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator^(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator&(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  // Forward declarations of non-member shift functions of (uintwide_t shift IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator<<(const uintwide_t<Digits2, LimbType>& u, const IntegralType n);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator>>(const uintwide_t<Digits2, LimbType>& u, const IntegralType n);

  // Forward declarations of non-member comparison functions of (uintwide_t cmp uintwide_t).
  template<const std::size_t Digits2, typename LimbType> bool operator==(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> bool operator!=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> bool operator> (const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> bool operator< (const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> bool operator>=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);
  template<const std::size_t Digits2, typename LimbType> bool operator<=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v);

  // Forward declarations of non-member comparison functions of (uintwide_t cmp IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator==(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator!=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator> (const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator< (const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator>=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator<=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v);

  // Forward declarations of non-member comparison functions of (IntegralType cmp uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator==(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator!=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator> (const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator< (const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator>=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator<=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v);

  #if defined(WIDE_INTEGER_DISABLE_IOSTREAM)
  #else

  // Forward declarations of I/O streaming functions.
  template<typename char_type,
           typename traits_type,
           const std::size_t Digits2,
           typename LimbType>
  std::basic_ostream<char_type,
                     traits_type>& operator<<(std::basic_ostream<char_type, traits_type>& out,
                                              const uintwide_t<Digits2, LimbType>& x);

  template<typename char_type,
           typename traits_type,
           const std::size_t Digits2,
           typename LimbType>
  std::basic_istream<char_type,
                     traits_type>& operator>>(std::basic_istream<char_type, traits_type>& in,
                                              uintwide_t<Digits2, LimbType>& x);

  #endif

  // Forward declarations of various number-theoretical tools.
  template<const std::size_t Digits2,
           typename LimbType>
  void swap(uintwide_t<Digits2, LimbType>& x,
            uintwide_t<Digits2, LimbType>& y);

  template<const std::size_t Digits2,
           typename LimbType>
  std::size_t lsb(const uintwide_t<Digits2, LimbType>& x);

  template<const std::size_t Digits2,
           typename LimbType>
  std::size_t msb(const uintwide_t<Digits2, LimbType>& x);

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> sqrt(const uintwide_t<Digits2, LimbType>& m);

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> cbrt(const uintwide_t<Digits2, LimbType>& m);

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> rootk(const uintwide_t<Digits2, LimbType>& m,
                                      const std::uint_fast8_t k);

  template<typename OtherUnsignedIntegralTypeP,
           const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> pow(const uintwide_t<Digits2, LimbType>& b,
                                    const OtherUnsignedIntegralTypeP&    p);

  template<typename OtherUnsignedIntegralTypeP,
           typename OtherUnsignedIntegralTypeM,
           const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> powm(const uintwide_t<Digits2, LimbType>& b,
                                     const OtherUnsignedIntegralTypeP&    p,
                                     const OtherUnsignedIntegralTypeM&    m);

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> gcd(const uintwide_t<Digits2, LimbType>& a,
                                    const uintwide_t<Digits2, LimbType>& b);

  template<const std::size_t Digits2,
           typename LimbType>
  class default_random_engine;

  template<const std::size_t Digits2,
           typename LimbType>
  class uniform_int_distribution;

  template<const std::size_t Digits2,
           typename LimbType>
  bool operator==(const uniform_int_distribution<Digits2, LimbType>& lhs,
                  const uniform_int_distribution<Digits2, LimbType>& rhs);

  template<const std::size_t Digits2,
           typename LimbType>
  bool operator!=(const uniform_int_distribution<Digits2, LimbType>& lhs,
                  const uniform_int_distribution<Digits2, LimbType>& rhs);

  template<typename DistributionType,
           typename GeneratorType,
           const std::size_t Digits2,
           typename LimbType>
  bool miller_rabin(const uintwide_t<Digits2, LimbType>& n,
                    const std::size_t                    number_of_trials,
                    DistributionType&                    distribution,
                    GeneratorType&                       generator);

  } } // namespace wide_integer::generic_template

  namespace std
  {
    // Forward declaration of specialization of std::numeric_limits<uintwide_t>.
    template<const std::size_t Digits2,
             typename LimbType>
    class numeric_limits<wide_integer::generic_template::uintwide_t<Digits2, LimbType>>;
  }

  namespace wide_integer { namespace generic_template { namespace detail {

  template<const std::size_t Digits2>
  struct verify_power_of_two
  {
    static const bool conditional_value =
      ((Digits2 != 0U) && ((Digits2 & (Digits2 - 1U)) == 0U));
  };

  // Helper templates for selecting integral types.
  template<const std::size_t BitCount> struct int_type_helper
  {
    static_assert((   ((BitCount >= 8U) && (BitCount <= 64U))
                   && (verify_power_of_two<BitCount>::conditional_value == true)),
                  "Error: int_type_helper is not intended to be used for this BitCount");

    using exact_unsigned_type = std::uintmax_t;
    using exact_signed_type   = std::intmax_t;
  };

  template<> struct int_type_helper< 8U> { using exact_unsigned_type = std::uint8_t;   using exact_signed_type = std::int8_t;   };
  template<> struct int_type_helper<16U> { using exact_unsigned_type = std::uint16_t;  using exact_signed_type = std::int16_t;  };
  template<> struct int_type_helper<32U> { using exact_unsigned_type = std::uint32_t;  using exact_signed_type = std::int32_t;  };
  template<> struct int_type_helper<64U> { using exact_unsigned_type = std::uint64_t;  using exact_signed_type = std::int64_t;  };

  // Use a local implementation of string copy.
  inline char* strcpy_unsafe(char* dst, const char* src)
  {
    while((*dst++ = *src++) != char('\0')) { ; }

    return dst;
  }

  // Use a local implementation of string length.
  inline std::size_t strlen_unsafe(const char* p_str)
  {
    const char* p_str_copy;

    for(p_str_copy = p_str; (*p_str_copy != char('\0')); ++p_str_copy) { ; }

    return std::size_t(p_str_copy - p_str);
  }

  template<typename ST,
           typename LT = typename detail::int_type_helper<std::size_t(std::numeric_limits<ST>::digits * 2)>::exact_unsigned_type>
  ST make_lo(const LT& u)
  {
    // From an unsigned integral input parameter of type LT,
    // extract the low part of it. The type of the extracted
    // low part is ST, which has half the width of LT.

    using local_ushort_type = ST;
    using local_ularge_type = LT;

    // Compile-time checks.
    static_assert((    (std::numeric_limits<local_ushort_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ularge_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ushort_type>::is_signed  == false)
                   &&  (std::numeric_limits<local_ularge_type>::is_signed  == false)
                   && ((std::numeric_limits<local_ushort_type>::digits * 2) == std::numeric_limits<local_ularge_type>::digits)),
                   "Error: Please check the characteristics of the template parameters ST and LT");

    return static_cast<local_ushort_type>(u);
  }

  template<typename ST,
           typename LT = typename detail::int_type_helper<std::size_t(std::numeric_limits<ST>::digits * 2)>::exact_unsigned_type>
  ST make_hi(const LT& u)
  {
    // From an unsigned integral input parameter of type LT,
    // extract the high part of it. The type of the extracted
    // high part is ST, which has half the width of LT.

    using local_ushort_type = ST;
    using local_ularge_type = LT;

    // Compile-time checks.
    static_assert((    (std::numeric_limits<local_ushort_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ularge_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ushort_type>::is_signed  == false)
                   &&  (std::numeric_limits<local_ularge_type>::is_signed  == false)
                   && ((std::numeric_limits<local_ushort_type>::digits * 2) == std::numeric_limits<local_ularge_type>::digits)),
                   "Error: Please check the characteristics of the template parameters ST and LT");

    return static_cast<local_ushort_type>(u >> std::numeric_limits<local_ushort_type>::digits);
  }

  template<typename ST,
           typename LT = typename detail::int_type_helper<std::size_t(std::numeric_limits<ST>::digits * 2)>::exact_unsigned_type>
  LT make_large(const ST& lo, const ST& hi)
  {
    // Create a composite unsigned integral value having type LT.
    // Two constituents are used having type ST, whereby the
    // width of ST is half the width of LT.

    using local_ushort_type = ST;
    using local_ularge_type = LT;

    // Compile-time checks.
    static_assert((    (std::numeric_limits<local_ushort_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ularge_type>::is_integer == true)
                   &&  (std::numeric_limits<local_ushort_type>::is_signed  == false)
                   &&  (std::numeric_limits<local_ularge_type>::is_signed  == false)
                   && ((std::numeric_limits<local_ushort_type>::digits * 2) == std::numeric_limits<local_ularge_type>::digits)),
                   "Error: Please check the characteristics of the template parameters ST and LT");

    return local_ularge_type(local_ularge_type(static_cast<local_ularge_type>(hi) << std::numeric_limits<ST>::digits) | lo);
  }

  template<typename UnsignedIntegralType>
  std::size_t lsb_helper(const UnsignedIntegralType& x)
  {
    // Compile-time checks.
    static_assert((   (std::numeric_limits<UnsignedIntegralType>::is_integer == true)
                   && (std::numeric_limits<UnsignedIntegralType>::is_signed  == false)),
                   "Error: Please check the characteristics of UnsignedIntegralType");

    using local_unsigned_integral_type = UnsignedIntegralType;

    std::size_t i;

    // This assumes that at least one bit is set.
    // Otherwise saturation of the index will occur.
    for(i = 0U; i < std::size_t(std::numeric_limits<local_unsigned_integral_type>::digits); ++i)
    {
      if((x & UnsignedIntegralType(local_unsigned_integral_type(1U) << i)) != 0U)
      {
        break;
      }
    }

    return i;
  }

  template<typename UnsignedIntegralType>
  std::size_t msb_helper(const UnsignedIntegralType& x)
  {
    // Compile-time checks.
    static_assert((   (std::numeric_limits<UnsignedIntegralType>::is_integer == true)
                   && (std::numeric_limits<UnsignedIntegralType>::is_signed  == false)),
                   "Error: Please check the characteristics of UnsignedIntegralType");

    using local_unsigned_integral_type = UnsignedIntegralType;

    std::ptrdiff_t i;

    // This assumes that at least one bit is set.
    // Otherwise underflow of the index will occur.
    for(i = std::ptrdiff_t(std::numeric_limits<local_unsigned_integral_type>::digits - 1); i >= 0; --i)
    {
      if((x & UnsignedIntegralType(local_unsigned_integral_type(1U) << i)) != 0U)
      {
        break;
      }
    }

    return std::size_t(i);
  }

  template<typename ST>
  ST integer_gcd_reduce_short(ST u, ST v)
  {
    // This implementation of GCD reduction is based on an
    // adaptation of existing code from Boost.Multiprecision.

    for(;;)
    {
      if(u > v)
      {
        std::swap(u, v);
      }

      if(u == v)
      {
        break;
      }

      v  -= u;
      v >>= detail::lsb_helper(v);
    }

    return u;
  }

  template<typename LT>
  LT integer_gcd_reduce_large(LT u, LT v)
  {
    // This implementation of GCD reduction is based on an
    // adaptation of existing code from Boost.Multiprecision.

    using local_ularge_type = LT;
    using local_ushort_type = typename detail::int_type_helper<std::size_t(std::numeric_limits<local_ularge_type>::digits / 2)>::exact_unsigned_type;

    for(;;)
    {
      if(u > v)
      {
        std::swap(u, v);
      }

      if(u == v)
      {
        break;
      }

      if(v <= local_ularge_type((std::numeric_limits<local_ushort_type>::max)()))
      {
        u = integer_gcd_reduce_short(local_ushort_type(v),
                                     local_ushort_type(u));

        break;
      }

      v -= u;

      while((std::uint_fast8_t(v) & 1U) == 0U)
      {
        v >>= 1;
      }
    }

    return u;
  }

  } } } // namespace wide_integer::generic_template::detail

  namespace wide_integer { namespace generic_template {

  template<const std::size_t Digits2,
           typename LimbType>
  class uintwide_t
  {
  public:
    // Verify that the Digits2 template parameter is a power of 2.
    static_assert(detail::verify_power_of_two<Digits2>::conditional_value == true,
                  "Error: The Digits2 template parameter must be a power of 2");

    // Class-local type definitions.
    using ushort_type = LimbType;
    using ularge_type = typename detail::int_type_helper<std::size_t(std::numeric_limits<ushort_type>::digits * 2)>::exact_unsigned_type;

    // More compile-time checks.
    static_assert((    (std::numeric_limits<ushort_type>::is_integer == true)
                   &&  (std::numeric_limits<ularge_type>::is_integer == true)
                   &&  (std::numeric_limits<ushort_type>::is_signed  == false)
                   &&  (std::numeric_limits<ularge_type>::is_signed  == false)
                   && ((std::numeric_limits<ushort_type>::digits * 2) == std::numeric_limits<ularge_type>::digits)),
                   "Error: Please check the characteristics of the template parameters ST and LT");

    // Helper constants for the digit characteristics.
    static const std::size_t my_digits   = Digits2;
    static const std::size_t my_digits10 = static_cast<int>((std::uintmax_t(my_digits) * UINTMAX_C(301)) / 1000U);

    // The number of limbs.
    static const std::size_t number_of_limbs =
      std::size_t(my_digits / std::size_t(std::numeric_limits<ushort_type>::digits));

    // The type of the internal data representation.
    using representation_type = std::array<ushort_type, number_of_limbs>;

    // The value type of the internal data representation.
    using value_type = typename representation_type::value_type;

    // The iterator types of the internal data representation.
    using iterator               = typename std::array<value_type, number_of_limbs>::iterator;
    using const_iterator         = typename std::array<value_type, number_of_limbs>::const_iterator;
    using reverse_iterator       = typename std::array<value_type, number_of_limbs>::reverse_iterator;
    using const_reverse_iterator = typename std::array<value_type, number_of_limbs>::const_reverse_iterator;

    // Types that have half or double the width of *this.
    using half_width_type   = uintwide_t<my_digits / 2U, ushort_type>;
    using double_width_type = uintwide_t<my_digits * 2U, ushort_type>;

    // Default destructor.
    ~uintwide_t() = default;

    // Default constructor.
    uintwide_t() = default;

    // Constructors from built-in unsigned integral types that
    // are less wide than ushort_type or exactly as wide as ushort_type.
    template<typename UnsignedIntegralType>
    uintwide_t(const UnsignedIntegralType v,
               typename std::enable_if<(   (std::is_fundamental<UnsignedIntegralType>::value == true)
                                        && (std::is_integral   <UnsignedIntegralType>::value == true)
                                        && (std::is_unsigned   <UnsignedIntegralType>::value == true)
                                        && (std::numeric_limits<UnsignedIntegralType>::digits <= std::numeric_limits<ushort_type>::digits))>::type* = nullptr)
    {
      values[0U] = ushort_type(v);

      std::fill(values.begin() + 1U, values.end(), ushort_type(0U));
    }

    // Constructors from built-in unsigned integral types that
    // are wider than ushort_type, and do not have exactly the
    // same width as ushort_type.
    template<typename UnsignedIntegralType>
    uintwide_t(const UnsignedIntegralType v,
               typename std::enable_if<(   (std::is_fundamental<UnsignedIntegralType>::value == true)
                                        && (std::is_integral   <UnsignedIntegralType>::value == true)
                                        && (std::is_unsigned   <UnsignedIntegralType>::value == true)
                                        && (std::numeric_limits<UnsignedIntegralType>::digits > std::numeric_limits<ushort_type>::digits))>::type* = nullptr)
    {
      std::uint_fast32_t right_shift_amount_v = 0U;
      std::uint_fast8_t  index_u              = 0U;

      for( ; (index_u < values.size()) && (right_shift_amount_v < std::uint_fast32_t(std::numeric_limits<UnsignedIntegralType>::digits)); ++index_u)
      {
        values[index_u] = ushort_type(v >> (int) right_shift_amount_v);

        right_shift_amount_v += std::uint_fast32_t(std::numeric_limits<ushort_type>::digits);
      }

      std::fill(values.begin() + index_u, values.end(), ushort_type(0U));
    }

    // Constructors from built-in signed integral types.
    template<typename SignedIntegralType>
    uintwide_t(const SignedIntegralType v,
               typename std::enable_if<(   (std::is_fundamental<SignedIntegralType>::value == true)
                                        && (std::is_integral   <SignedIntegralType>::value == true)
                                        && (std::is_signed     <SignedIntegralType>::value == true))>::type* = nullptr)
    {
      using local_signed_integral_type   = SignedIntegralType;
      using local_unsigned_integral_type = typename detail::int_type_helper<std::numeric_limits<local_signed_integral_type>::digits + 1>::exact_unsigned_type;

      const bool is_neg = (v < local_signed_integral_type(0));

      const local_unsigned_integral_type u =
        ((is_neg == false) ? local_unsigned_integral_type(v) : local_unsigned_integral_type(-v));

      operator=(uintwide_t(u));

      if(is_neg) { negate(); }
    }

    // Constructor from the internal data representation.
    uintwide_t(const representation_type& other_rep)
    {
      std::copy(other_rep.cbegin(), other_rep.cend(), values.begin());
    }

    // Constructor from a C-style array.
    template<const std::size_t N>
    uintwide_t(const ushort_type(&init)[N])
    {
      static_assert(N <= number_of_limbs,
                    "Error: The initialization list has too many elements.");

      std::copy(init, init + (std::min)(N, number_of_limbs), values.begin());
    }

    // Copy constructor.
    uintwide_t(const uintwide_t& other) : values(other.values) { }

    // Constructor from the double-width type.
    // This constructor is explicit because it
    // is a narrowing conversion.
    template<typename UnknownUnsignedWideIntegralType = double_width_type>
    explicit uintwide_t(const UnknownUnsignedWideIntegralType& v,
                        typename std::enable_if<(   (std::is_same<UnknownUnsignedWideIntegralType, double_width_type>::value == true)
                                                 && (128U <= my_digits))>::type* = nullptr)
    {
      std::copy(v.crepresentation().cbegin(),
                v.crepresentation().cbegin() + (v.crepresentation().size() / 2U),
                values.begin());
    }

    // Constructor from a constant character string.
    uintwide_t(const char* str_input)
    {
      if(rd_string(str_input) == false)
      {
        std::fill(values.begin(), values.end(), (std::numeric_limits<ushort_type>::max)());
      }
    }

    // Move constructor.
    uintwide_t(uintwide_t&& other) : values(static_cast<representation_type&&>(other.values)) { }

    // Assignment operator.
    uintwide_t& operator=(const uintwide_t& other)
    {
      if(this != &other)
      {
        std::copy(other.values.cbegin(), other.values.cend(), values.begin());
      }

      return *this;
    }

    // Trivial move assignment operator.
    uintwide_t& operator=(uintwide_t&& other)
    {
      std::copy(other.values.cbegin(), other.values.cend(), values.begin());

      return *this;
    }

    // Implement a cast operator that casts to any
    // built-in signed or unsigned integral type.
    template<typename UnknownBuiltInIntegralType,
             typename = typename std::enable_if<
                          (   (std::is_fundamental<UnknownBuiltInIntegralType>::value == true)
                           && (std::is_integral   <UnknownBuiltInIntegralType>::value == true))>::type>
    explicit operator UnknownBuiltInIntegralType() const
    {
      using local_unknown_integral_type  = UnknownBuiltInIntegralType;

      using local_unsigned_integral_type =
        typename detail::int_type_helper<
          std::numeric_limits<local_unknown_integral_type>::is_signed
            ? std::numeric_limits<local_unknown_integral_type>::digits + 1
            : std::numeric_limits<local_unknown_integral_type>::digits + 0>::exact_unsigned_type;

      local_unsigned_integral_type cast_result;

      const std::uint_fast8_t digits_ratio = 
        std::uint_fast8_t(  std::numeric_limits<local_unsigned_integral_type>::digits
                          / std::numeric_limits<value_type>::digits);

      switch(digits_ratio)
      {
        case 0:
        case 1:
          // The input parameter is less wide or equally as wide as the limb width.
          cast_result = static_cast<local_unsigned_integral_type>(values[0U]);
          break;

        default:
          // The input parameter is wider than the limb width.
          cast_result = 0U;

          for(std::uint_fast8_t i = 0U; i < digits_ratio; ++i)
          {
            const local_unsigned_integral_type u =
              local_unsigned_integral_type(values[i]) << (std::numeric_limits<value_type>::digits * int(i));

            cast_result |= u;
          }
          break;
      }

      return local_unknown_integral_type(cast_result);
    }

    // Implement the cast operator that casts to the double-width type.
    template<typename UnknownUnsignedWideIntegralType = double_width_type,
             typename = typename std::enable_if<(   (std::is_same<UnknownUnsignedWideIntegralType, double_width_type>::value == true)
                                                 && (my_digits >= 128U))>::type>
    operator double_width_type() const
    {
      double_width_type local_double_width_instance;

      std::copy(values.cbegin(),
                values.cend(),
                local_double_width_instance.representation().begin());

      std::fill(local_double_width_instance.representation().begin() + number_of_limbs,
                local_double_width_instance.representation().end(),
                ushort_type(0U));

      return local_double_width_instance;
    }

    // Intentionally delete the cast operator that casts
    // to the half-width type. This cast is deleted because
    // it is a narrowing conversion. There is an explicit
    // constructor above for this conversion.
    template<typename UnknownUnsignedWideIntegralType = half_width_type,
             typename = typename std::enable_if<
                          std::is_same<UnknownUnsignedWideIntegralType,
                                       half_width_type>::value == true>::type>
    operator half_width_type() const = delete;

    // Provide a user interface to the internal data representation.
          representation_type&  representation()       { return values; }
    const representation_type&  representation() const { return values; }
    const representation_type& crepresentation() const { return values; }

    // Unary operators: not, plus and minus.
    uintwide_t& operator+() const { return *this; }
    uintwide_t  operator-() const { uintwide_t tmp(*this); tmp.negate(); return tmp; }

    uintwide_t& operator+=(const uintwide_t& other)
    {
      // Unary addition function.
      std::array<ularge_type, number_of_limbs> result_as_ularge_array;

      ushort_type carry(0U);

      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        result_as_ularge_array[i] = ularge_type(ularge_type(values[i]) + other.values[i]) + carry;

        carry = detail::make_hi<ushort_type>(result_as_ularge_array[i]);

        values[i] = ushort_type(result_as_ularge_array[i]);
      }

      return *this;
    }

    uintwide_t& operator-=(const uintwide_t& other)
    {
      // Unary subtraction function.
      std::array<ularge_type, number_of_limbs> result_as_ularge_array;

      bool has_borrow = false;

      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        result_as_ularge_array[i] = ularge_type(values[i]) - other.values[i];

        if(has_borrow) { --result_as_ularge_array[i]; }

        has_borrow = (detail::make_hi<ushort_type>(result_as_ularge_array[i]) != ushort_type(0U));

        values[i] = ushort_type(result_as_ularge_array[i]);
      }

      return *this;
    }

    uintwide_t& operator*=(const uintwide_t& other)
    {
      // Unary multiplication function.
      std::array<ushort_type, number_of_limbs> result = {{ 0U }};

      multiplication_loop_schoolbook(values.data(),
                                     other.values.data(),
                                     result.data(),
                                     result.size());

      values = result;

      return *this;
    }

    uintwide_t& operator/=(const uintwide_t& other)
    {
      // Unary division function.
      quotient_and_remainder_knuth(other, nullptr);

      return *this;
    }

    uintwide_t& operator%=(const uintwide_t& other)
    {
      // Unary modulus function.
      uintwide_t remainder;

      quotient_and_remainder_knuth(other, &remainder);

      values = remainder.values;

      return *this;
    }

    // Operators pre-increment and pre-decrement.
    uintwide_t& operator++() { preincrement(); return *this; }
    uintwide_t& operator--() { predecrement(); return *this; }

    // Operators post-increment and post-decrement.
    uintwide_t operator++(int) { const uintwide_t w(*this); preincrement(); return w; }
    uintwide_t operator--(int) { const uintwide_t w(*this); predecrement(); return w; }

    uintwide_t& operator~()
    {
      // Bitwise NOT.
      bitwise_not();

      return *this;
    }

    uintwide_t& operator|=(const uintwide_t& other)
    {
      // Bitwise OR.
      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        values[i] |= other.values[i];
      }

      return *this;
    }

    uintwide_t& operator^=(const uintwide_t& other)
    {
      // Bitwise XOR.
      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        values[i] ^= other.values[i];
      }

      return *this;
    }

    uintwide_t& operator&=(const uintwide_t& other)
    {
      // Bitwise AND.
      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        values[i] &= other.values[i];
      }

      return *this;
    }

    template<typename IntegralType>
    typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                             && (std::is_integral   <IntegralType>::value == true)), uintwide_t>::type&
    operator<<=(const IntegralType n)
    {
      // Left-shift operator.
      if     (n <  0) { operator>>=(n); }
      else if(n == 0) { ; }
      else
      {
        if(std::size_t(n) >= my_digits)
        {
          std::fill(values.begin(), values.end(), value_type(0U));
        }
        else
        {
          const std::size_t offset            = std::size_t(n) / std::size_t(std::numeric_limits<ushort_type>::digits);
          const std::size_t left_shift_amount = std::size_t(n) % std::size_t(std::numeric_limits<ushort_type>::digits);

          std::copy_backward(values.data(),
                             values.data() + (number_of_limbs - offset),
                             values.data() +  number_of_limbs);

          std::fill(values.begin(), values.begin() + offset, ushort_type(0U));

          ushort_type part_from_previous_value = ushort_type(0U);

          using local_integral_type = IntegralType;

          if(left_shift_amount != local_integral_type(0U))
          {
            for(std::size_t i = offset; i < number_of_limbs; ++i)
            {
              const ushort_type t = values[i];

              values[i] = (t << local_integral_type(left_shift_amount)) | part_from_previous_value;

              part_from_previous_value = ushort_type(t >> local_integral_type(std::size_t(std::numeric_limits<ushort_type>::digits - left_shift_amount)));
            }
          }
        }
      }

      return *this;
    }

    template<typename IntegralType>
    typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                             && (std::is_integral   <IntegralType>::value == true)), uintwide_t>::type&
    operator>>=(const IntegralType n)
    {
      // Right-shift operator.
      if     (n <  0) { operator<<=(n); }
      else if(n == 0) { ; }
      else
      {
        if(std::size_t(n) >= my_digits)
        {
          std::fill(values.begin(), values.end(), value_type(0U));
        }
        else
        {
          const std::size_t offset             = std::size_t(n) / std::size_t(std::numeric_limits<ushort_type>::digits);
          const std::size_t right_shift_amount = std::size_t(n) % std::size_t(std::numeric_limits<ushort_type>::digits);

          std::copy(values.begin() + offset,
                    values.begin() + number_of_limbs,
                    values.begin());

          std::fill(values.rbegin(), values.rbegin() + std::ptrdiff_t(offset), ushort_type(0U));

          ushort_type part_from_previous_value = ushort_type(0U);

          using local_integral_type = IntegralType;

          if(right_shift_amount != local_integral_type(0U))
          {
            for(std::size_t i = ((number_of_limbs - 1U) - offset); std::ptrdiff_t(i) >= 0; --i)
            {
              const ushort_type t = values[i];

              values[i] = (t >> local_integral_type(right_shift_amount)) | part_from_previous_value;

              part_from_previous_value = ushort_type(t << local_integral_type(std::size_t(std::numeric_limits<ushort_type>::digits - right_shift_amount)));
            }
          }
        }
      }

      return *this;
    }

    // Implement comparison operators.
    bool operator==(const uintwide_t& other) const { return (compare(other) == std::int_fast8_t( 0)); }
    bool operator< (const uintwide_t& other) const { return (compare(other) == std::int_fast8_t(-1)); }
    bool operator> (const uintwide_t& other) const { return (compare(other) == std::int_fast8_t( 1)); }
    bool operator!=(const uintwide_t& other) const { return (compare(other) != std::int_fast8_t( 0)); }
    bool operator<=(const uintwide_t& other) const { return (compare(other) <= std::int_fast8_t( 0)); }
    bool operator>=(const uintwide_t& other) const { return (compare(other) >= std::int_fast8_t( 0)); }

    // Helper functions for supporting std::numeric_limits<>.
    static uintwide_t limits_helper_max()
    {
      uintwide_t val;

      std::fill(val.values.begin(),
                val.values.end(),
                (std::numeric_limits<ushort_type>::max)());

      return val;
    }

    static uintwide_t limits_helper_min()
    {
      return uintwide_t(std::uint8_t(0U));
    }

    // Define the maximum buffer sizes for extracting
    // octal, decimal and hexadecimal string representations.
    static const std::size_t wr_string_max_buffer_size_oct = (16U + (my_digits / 3U)) + std::size_t(((my_digits % 3U) != 0U) ? 1U : 0U) + 1U;
    static const std::size_t wr_string_max_buffer_size_hex = (32U + (my_digits / 4U)) + 1U;
    static const std::size_t wr_string_max_buffer_size_dec = (20U + std::size_t((std::uintmax_t(my_digits) * UINTMAX_C(301)) / UINTMAX_C(1000))) + 1U;

    // Write string function.
    bool wr_string(      char*             str_result,
                   const std::uint_fast8_t base_rep     = 0x10U,
                   const bool              show_base    = true,
                   const bool              show_pos     = false,
                   const bool              is_uppercase = true,
                         std::size_t       field_width  = 0U,
                   const char              fill_char    = char('0')) const
    {
      uintwide_t t(*this);

      bool wr_string_is_ok = true;

      if(base_rep == 8U)
      {
        const ushort_type mask(std::uint8_t(0x7U));

        char str_temp[wr_string_max_buffer_size_oct];

        std::size_t pos = (sizeof(str_temp) - 1U);

        if(t.is_zero())
        {
          --pos;

          str_temp[pos] = char('0');
        }
        else
        {
          while(t.is_zero() == false)
          {
            char c = char(t.values[0U] & mask);

            if(c <= 8) { c += char(0x30); }

            --pos;

            str_temp[pos] = c;

            t >>= 3;
          }
        }

        if(show_base)
        {
          --pos;

          str_temp[pos] = char('0');
        }

        if(show_pos)
        {
          --pos;

          str_temp[pos] = char('+');
        }

        if(field_width != 0U)
        {
          field_width = (std::min)(field_width, std::size_t(sizeof(str_temp) - 1U));

          while(std::ptrdiff_t(pos) > std::ptrdiff_t((sizeof(str_temp) - 1U) - field_width))
          {
            --pos;

            str_temp[pos] = fill_char;
          }
        }

        str_temp[std::size_t(sizeof(str_temp) - 1U)] = char('\0');

        detail::strcpy_unsafe(str_result, str_temp + pos);
      }
      else if(base_rep == 10U)
      {
        char str_temp[wr_string_max_buffer_size_dec];

        std::size_t pos = (sizeof(str_temp) - 1U);

        if(t.is_zero() == true)
        {
          --pos;

          str_temp[pos] = char('0');
        }
        else
        {
          const uintwide_t ten(std::uint8_t(10U));

          while(t.is_zero() == false)
          {
            const uintwide_t t_temp(t);

            t /= ten;

            char c = char(ushort_type((t_temp - (t * ten)).values[0U]));

            if(c <= char(9)) { c += char(0x30); }

            --pos;

            str_temp[pos] = c;
          }
        }

        if(show_pos)
        {
          --pos;

          str_temp[pos] = char('+');
        }

        if(field_width != 0U)
        {
          field_width = (std::min)(field_width, std::size_t(sizeof(str_temp) - 1U));

          while(std::ptrdiff_t(pos) > std::ptrdiff_t((sizeof(str_temp) - 1U) - field_width))
          {
            --pos;

            str_temp[pos] = fill_char;
          }
        }

        str_temp[std::size_t(sizeof(str_temp) - 1U)] = char('\0');

        detail::strcpy_unsafe(str_result, str_temp + pos);
      }
      else if(base_rep == 16U)
      {
        const ushort_type mask(std::uint8_t(0xFU));

        char str_temp[wr_string_max_buffer_size_hex];

        std::size_t pos = (sizeof(str_temp) - 1U);

        if(t.is_zero() == true)
        {
          --pos;

          str_temp[pos] = char('0');
        }
        else
        {
          while(t.is_zero() == false)
          {
            char c(t.values[0U] & mask);

            if      (c <= char(  9))                      { c += char(0x30); }
            else if((c >= char(0xA)) && (c <= char(0xF))) { c += (is_uppercase ? char(55) : char(87)); }

            --pos;

            str_temp[pos] = c;

            t >>= 4;
          }
        }

        if(show_base)
        {
          --pos;

          str_temp[pos] = (is_uppercase ? char('X') : char('x'));

          --pos;

          str_temp[pos] = char('0');
        }

        if(show_pos)
        {
          --pos;

          str_temp[pos] = char('+');
        }

        if(field_width != 0U)
        {
          field_width = (std::min)(field_width, std::size_t(sizeof(str_temp) - 1U));

          while(std::ptrdiff_t(pos) > std::ptrdiff_t((sizeof(str_temp) - 1U) - field_width))
          {
            --pos;

            str_temp[pos] = fill_char;
          }
        }

        str_temp[std::size_t(sizeof(str_temp) - 1U)] = char('\0');

        detail::strcpy_unsafe(str_result, str_temp + pos);
      }
      else
      {
        wr_string_is_ok = false;
      }

      return wr_string_is_ok;
    }

  private:
    representation_type values;

    // Read string function.
    bool rd_string(const char* str_input)
    {
      std::fill(values.begin(), values.end(), ushort_type(0U));

      const std::size_t str_length = detail::strlen_unsafe(str_input);

      std::uint_fast8_t base = 10U;

      std::size_t pos = 0U;

      // Skip over a potential plus sign.
      if((str_length > 0U) && (str_input[0U] == char('+')))
      {
        ++pos;
      }

      // Perform a dynamic detection of the base.
      if(str_length > (pos + 0U))
      {
        const bool might_be_oct_or_hex = ((str_input[pos + 0U] == char('0')) && (str_length > (pos + 1U)));

        if(might_be_oct_or_hex)
        {
          if((str_input[pos + 1U] >= char('0')) && (str_input[pos + 1U] <= char('8')))
          {
            // The input format is octal.
            base = 8U;

            pos += 1U;
          }
          else if((str_input[pos + 1U] == char('x')) || (str_input[pos + 1U] == char('X')))
          {
            // The input format is hexadecimal.
            base = 16U;

            pos += 2U;
          }
        }
        else if((str_input[pos + 0U] >= char('0')) && (str_input[pos + 0U] <= char('9')))
        {
          // The input format is decimal.
          ;
        }
      }

      bool char_is_valid = true;

      for( ; ((pos < str_length) && char_is_valid); ++pos)
      {
        std::uint8_t c = std::uint8_t(str_input[pos]);

        const bool char_is_apostrophe = (c == char(39));

        if(char_is_apostrophe == false)
        {
          if(base == 8U)
          {
            if  ((c >= char('0')) && (c <= char('8'))) { c -= std::uint8_t(0x30U); }
            else                                       { char_is_valid = false; }

            if(char_is_valid)
            {
              operator<<=(3);

              values[0U] |= std::uint8_t(c);
            }
          }
          else if(base == 10U)
          {
            if   ((c >= std::uint8_t('0')) && (c <= std::uint8_t('9'))) { c -= std::uint8_t(0x30U); }
            else                                                        { char_is_valid = false; }

            if(char_is_valid)
            {
              operator*=(10U);

              operator+=(c);
            }
          }
          else if(base == 16U)
          {
            if     ((c >= std::uint8_t('a')) && (c <= std::uint8_t('f'))) { c -= std::uint8_t(  87U); }
            else if((c >= std::uint8_t('A')) && (c <= std::uint8_t('F'))) { c -= std::uint8_t(  55U); }
            else if((c >= std::uint8_t('0')) && (c <= std::uint8_t('9'))) { c -= std::uint8_t(0x30U); }
            else                                                          { char_is_valid = false; }

            if(char_is_valid)
            {
              operator<<=(4);

              values[0U] |= c;
            }
          }
        }
      }

      return char_is_valid;
    }

    static void multiplication_loop_schoolbook(      ushort_type* pu,
                                               const ushort_type* pv,
                                                     ushort_type* pw,
                                               const std::size_t  count)
    {
      for(std::size_t j = 0U; j < count; ++j)
      {
        if(pv[j] != ushort_type(0U))
        {
          ushort_type carry = ushort_type(0U);

          for(std::size_t i = 0U, iplusj = i + j; iplusj < count; ++i, ++iplusj)
          {
            const ularge_type t =
              ularge_type(ularge_type(ularge_type(pu[i]) * pv[j]) + pw[iplusj]) + carry;

            pw[iplusj] = detail::make_lo<ushort_type>(t);
            carry      = detail::make_hi<ushort_type>(t);
          }
        }
      }
    }

    void quotient_and_remainder_knuth(const uintwide_t& other, uintwide_t* remainder)
    {
      // TBD: Consider cleaning up the unclear flow-control
      // caused by numerous return statements in this subroutine.

      // Use Knuth's long division algorithm.
      // The loop-ordering of indexes in Knuth's original
      // algorithm has been reversed due to the data format
      // used here.

      // See also:
      // D.E. Knuth, "The Art of Computer Programming, Volume 2:
      // Seminumerical Algorithms", Addison-Wesley (1998),
      // Section 4.3.1 Algorithm D and Exercise 16.

      using local_uint_index_type = std::size_t;

      local_uint_index_type u_offset = local_uint_index_type(0U);
      local_uint_index_type v_offset = local_uint_index_type(0U);

      // Compute the offsets for u and v.
      for(local_uint_index_type i = 0U; (i < number_of_limbs) && (      values[(number_of_limbs - 1U) - i] == ushort_type(0U)); ++i) { ++u_offset; }
      for(local_uint_index_type i = 0U; (i < number_of_limbs) && (other.values[(number_of_limbs - 1U) - i] == ushort_type(0U)); ++i) { ++v_offset; }

      if(v_offset == local_uint_index_type(number_of_limbs))
      {
        // The denominator is zero. Set the maximum value and return.
        // This also catches (0 / 0) and sets the maximum value for it.
        operator=(limits_helper_max());

        if(remainder != nullptr)
        {
          *remainder = uintwide_t(std::uint8_t(0U));
        }

        return;
      }

      if(u_offset == local_uint_index_type(number_of_limbs))
      {
        // The numerator is zero. Do nothing and return.

        if(remainder != nullptr)
        {
          *remainder = uintwide_t(std::uint8_t(0U));
        }

        return;
      }

      {
        const int result_of_compare_left_with_right = compare(other);

        const bool left_is_less_than_right = (result_of_compare_left_with_right == -1);
        const bool left_is_equal_to_right  = (result_of_compare_left_with_right ==  0);

        if(left_is_less_than_right)
        {
          // If the denominator is larger than the numerator,
          // then the result of the division is zero.
          if(remainder != nullptr)
          {
            *remainder = *this;
          }

          operator=(std::uint8_t(0U));

          return;
        }

        if(left_is_equal_to_right)
        {
          // If the denominator is equal to the numerator,
          // then the result of the division is one.
          operator=(std::uint8_t(1U));

          if(remainder != nullptr)
          {
            *remainder = uintwide_t(std::uint8_t(0U));
          }

          return;
        }
      }

      if(v_offset == local_uint_index_type(number_of_limbs - 1U))
      {
        // The denominator has one single limb.
        // Use a one-dimensional division algorithm.

              ularge_type long_numerator    = ularge_type(0U);
        const ushort_type short_denominator = other.values[0U];

        ushort_type hi_part = ushort_type(0U);

        for(local_uint_index_type i = local_uint_index_type((number_of_limbs - 1U) - u_offset); std::ptrdiff_t(i) >= 0; --i)
        {
          long_numerator = ularge_type(values[i]) + ((long_numerator - ularge_type(ularge_type(short_denominator) * hi_part)) << std::numeric_limits<ushort_type>::digits);

          values[i] = detail::make_lo<ushort_type>(ularge_type(long_numerator / short_denominator));

          hi_part = values[i];
        }

        if(remainder != nullptr)
        {
          long_numerator = ularge_type(values[0U]) + ((long_numerator - ularge_type(ularge_type(short_denominator) * hi_part)) << std::numeric_limits<ushort_type>::digits);

          *remainder = ushort_type(long_numerator >> std::numeric_limits<ushort_type>::digits);

          if(u_offset != 0U)
          {
            std::fill(values.begin() + std::size_t((number_of_limbs - 1U) - u_offset),
                      values.end(),
                      ushort_type(0U));
          }
        }

        return;
      }

      // We will now use the Knuth long division algorithm.
      {
        // Compute the normalization factor d.
        const ularge_type d_large =
          ularge_type(  ((ularge_type(std::uint8_t(1U))) << std::numeric_limits<ushort_type>::digits)
                      /   ularge_type(ularge_type(other.values[(number_of_limbs - 1U) - v_offset]) + ushort_type(1U)));

        const ushort_type d = detail::make_lo<ushort_type>(d_large);

        // Step D1(b), normalize u -> u * d = uu.
        // Note the added digit in uu and also that
        // the data of uu have not been initialized yet.

        std::array<ushort_type, number_of_limbs + 1U> uu;

        if(d == ushort_type(1U))
        {
          // The normalization is one.
          std::copy(values.cbegin(), values.cend(), uu.begin());

          uu.back() = ushort_type(0U);
        }
        else
        {
          // Multiply u by d.
          ushort_type carry = 0U;

          local_uint_index_type i;

          for(i = local_uint_index_type(0U); i < local_uint_index_type(number_of_limbs - u_offset); ++i)
          {
            const ularge_type t = ularge_type(ularge_type(values[i]) * d) + carry;

            uu[i] = detail::make_lo<ushort_type>(t);
            carry = detail::make_hi<ushort_type>(t);
          }

          uu[i] = carry;
        }

        std::array<ushort_type, number_of_limbs> vv;

        // Step D1(c): normalize v -> v * d = vv.
        if(d == ushort_type(1U))
        {
          // The normalization is one.
          vv = other.values;
        }
        else
        {
          // Multiply v by d.
          ushort_type carry = 0U;

          for(local_uint_index_type i = local_uint_index_type(0U); i < local_uint_index_type(number_of_limbs - v_offset); ++i)
          {
            const ularge_type t = ularge_type(ularge_type(other.values[i]) * d) + carry;

            vv[i] = detail::make_lo<ushort_type>(t);
            carry = detail::make_hi<ushort_type>(t);
          }
        }

        // Step D2: Initialize j.
        // Step D7: Loop on j from m to 0.

        const local_uint_index_type n = local_uint_index_type(number_of_limbs - v_offset);
        const local_uint_index_type m = local_uint_index_type(number_of_limbs - u_offset) - n;

        for(local_uint_index_type j = local_uint_index_type(0U); j <= m; ++j)
        {
          // Step D3 [Calculate q_hat].
          //   if u[j] == v[j0]
          //     set q_hat = b - 1
          //   else
          //     set q_hat = (u[j] * b + u[j + 1]) / v[1]

          const local_uint_index_type uj     = (((number_of_limbs + 1U) - 1U) - u_offset) - j;
          const local_uint_index_type vj0    =   (number_of_limbs       - 1U) - v_offset;
          const ularge_type           u_j_j1 = (ularge_type(uu[uj]) << std::numeric_limits<ushort_type>::digits) + uu[uj - 1U];

          ularge_type q_hat = ((uu[uj] == vv[vj0])
                                ? ularge_type((std::numeric_limits<ushort_type>::max)())
                                : u_j_j1 / ularge_type(vv[vj0]));

          // Decrease q_hat if necessary.
          // This means that q_hat must be decreased if the
          // expression [(u[uj] * b + u[uj - 1] - q_hat * v[vj0 - 1]) * b]
          // exceeds the range of uintwide_t.

          ularge_type t;

          for(;;)
          {
            t = u_j_j1 - ularge_type(q_hat * ularge_type(vv[vj0]));

            if(detail::make_hi<ushort_type>(t) != ushort_type(0U))
            {
              break;
            }

            if(   ularge_type(ularge_type(vv[vj0 - 1U]) * q_hat)
               <= ularge_type((t << std::numeric_limits<ushort_type>::digits) + uu[uj - 2U]))
            {
              break;
            }

            --q_hat;
          }

          // Step D4: Multiply and subtract.
          // Replace u[j, ... j + n] by u[j, ... j + n] - q_hat * v[1, ... n].

          // Set nv = q_hat * (v[1, ... n]).
          {
            std::array<ushort_type, number_of_limbs + 1U> nv;

            ushort_type carry = 0U;

            local_uint_index_type i;

            for(i = local_uint_index_type(0U); i < n; ++i)
            {
              t     = ularge_type(ularge_type(vv[i]) * q_hat) + carry;
              nv[i] = detail::make_lo<ushort_type>(t);
              carry = detail::make_hi<ushort_type>(t);
            }

            nv[i] = carry;

            {
              // Subtract nv[0, ... n] from u[j, ... j + n].
              std::uint_fast8_t     borrow = 0U;
              local_uint_index_type ul     = uj - n;

              for(i = local_uint_index_type(0U); i <= n; ++i, ++ul)
              {
                t      = ularge_type(ularge_type(uu[ul]) - nv[i]) - ushort_type(borrow);
                uu[ul] =   detail::make_lo<ushort_type>(t);
                borrow = ((detail::make_hi<ushort_type>(t) != ushort_type(0U)) ? 1U : 0U);
              }

              // Get the result data.
              values[local_uint_index_type(m - j)] = detail::make_lo<ushort_type>(q_hat);

              // Step D5: Test the remainder.
              // Set the result value: Set result.m_data[m - j] = q_hat.
              // Use the condition (u[j] < 0), in other words if the borrow
              // is non-zero, then step D6 needs to be carried out.

              if(borrow != std::uint_fast8_t(0U))
              {
                // Step D6: Add back.
                // Add v[1, ... n] back to u[j, ... j + n],
                // and decrease the result by 1.

                carry = 0U;
                ul    = uj - n;

                for(i = local_uint_index_type(0U); i < n; ++i, ++ul)
                {
                  t      = ularge_type(ularge_type(uu[ul]) + vv[i]) + carry;
                  uu[ul] = detail::make_lo<ushort_type>(t);
                  carry  = detail::make_hi<ushort_type>(t);
                }

                // A potential test case for uint512_t is:
                //   QuotientRemainder
                //     [698937339790347543053797400564366118744312537138445607919548628175822115805812983955794321304304417541511379093392776018867245622409026835324102460829431,
                //      100041341335406267530943777943625254875702684549707174207105689918734693139781]
                //
                //     {6986485091668619828842978360442127600954041171641881730123945989288792389271,
                //      100041341335406267530943777943625254875702684549707174207105689918734693139780}

                --values[local_uint_index_type(m - j)];
              }
            }
          }
        }

        // Clear the data elements that have not
        // been computed in the division algorithm.
        std::fill(values.begin() + (m + 1U), values.end(), ushort_type(0U));

        if(remainder != nullptr)
        {
          if(d == 1)
          {
            std::copy(uu.cbegin(),
                      uu.cbegin() + (number_of_limbs - v_offset),
                      remainder->values.begin());
          }
          else
          {
            ushort_type previous_u = ushort_type(0U);

            for(local_uint_index_type rl = n - 1U, ul = local_uint_index_type(number_of_limbs - (v_offset + 1U)); std::ptrdiff_t(rl) >= 0; --rl, --ul)
            {
              const ularge_type t = ularge_type(uu[ul] + ularge_type(ularge_type(previous_u) << std::numeric_limits<ushort_type>::digits));

              remainder->values[rl] = detail::make_lo<ushort_type>(ularge_type(t / d));
              previous_u            = ushort_type(t - ularge_type(ularge_type(d) * remainder->values[rl]));
            }
          }

          std::fill(remainder->values.begin() + n,
                    remainder->values.end(),
                    ushort_type(0U));
        }
      }
    }

    std::int_fast8_t compare(const uintwide_t& other) const
    {
      std::int_fast8_t return_value;
      std::ptrdiff_t   element_index;

      for(element_index = std::ptrdiff_t(number_of_limbs - 1U); element_index >= std::ptrdiff_t(0); --element_index)
      {
        if(values[std::size_t(element_index)] != other.values[std::size_t(element_index)])
        {
          break;
        }
      }

      if(element_index == std::ptrdiff_t(-1))
      {
        return_value = std::int_fast8_t(0);
      }
      else
      {
        const bool left_is_greater_than_right =
          (values[std::size_t(element_index)] > other.values[std::size_t(element_index)]);

        return_value = (left_is_greater_than_right ? std::int_fast8_t(1) : std::int_fast8_t(-1));
      }

      return return_value;
    }

    void bitwise_not()
    {
      for(std::size_t i = 0U; i < number_of_limbs; ++i)
      {
        values[i] = value_type(~values[i]);
      }
    }

    void preincrement()
    {
      // Implement pre-increment.
      std::size_t i = 0U;

      for( ; (i < (values.size() - 1U)) && (++values[i] == value_type(0U)); ++i)
      {
        ;
      }

      if(i == (values.size() - 1U))
      {
        ++values[i];
      }
    }

    void predecrement()
    {
      // Implement pre-decrement.
      std::size_t i = 0U;

      for( ; (i < (values.size() - 1U)) && (values[i]-- == value_type(0U)); ++i)
      {
        ;
      }

      if(i == (values.size() - 1U))
      {
        --values[i];
      }
    }

    void negate()
    {
      bitwise_not();

      preincrement();
    }

    bool is_zero() const
    {
      return std::all_of(values.cbegin(),
                         values.cend(),
                         [](const value_type& u) -> bool
                         {
                           return (u == value_type(0U));
                         });
    }
  };

  // Define some convenient unsigned wide integer types.
  using uint64_t   = uintwide_t<  64U, std::uint16_t>;
  using uint128_t  = uintwide_t< 128U>;
  using uint256_t  = uintwide_t< 256U>;
  using uint512_t  = uintwide_t< 512U>;
  using uint1024_t = uintwide_t<1024U>;
  using uint2048_t = uintwide_t<2048U>;
  using uint4096_t = uintwide_t<4096U>;
  using uint8192_t = uintwide_t<8192U>;

  // Insert a base class for numeric_limits<> support.
  // This class inherits from std::numeric_limits<unsigned int>
  // in order to provide limits for a non-specific unsigned type.

  template<typename WideUnsignedIntegerType>
  class numeric_limits_uintwide_t_base : public std::numeric_limits<unsigned int>
  {
  private:
    using local_wide_integer_type = WideUnsignedIntegerType;

  public:
    static const int digits   = static_cast<int>(local_wide_integer_type::my_digits);
    static const int digits10 = static_cast<int>(local_wide_integer_type::my_digits10);

    static local_wide_integer_type (max)() { return local_wide_integer_type::limits_helper_max(); }
    static local_wide_integer_type (min)() { return local_wide_integer_type::limits_helper_min(); }
  };

  } } // namespace wide_integer::generic_template

  namespace std
  {
    // Specialization of std::numeric_limits<uintwide_t>.
    template<const std::size_t Digits2,
             typename LimbType>
    class numeric_limits<wide_integer::generic_template::uintwide_t<Digits2, LimbType>>
      : public wide_integer::generic_template::numeric_limits_uintwide_t_base<wide_integer::generic_template::uintwide_t<Digits2, LimbType>> { };
  }

  namespace wide_integer { namespace generic_template {

  // Non-member binary add, sub, mul, div, mod of (uintwide_t op uintwide_t).
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator+ (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator+=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator- (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator-=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator* (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator*=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator/ (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator/=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator% (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator%=(right); }

  // Non-member binary logic operations of (uintwide_t op uintwide_t).
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator| (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator|=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator^ (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator^=(right); }
  template<const std::size_t Digits2, typename LimbType> uintwide_t<Digits2, LimbType> operator& (const uintwide_t<Digits2, LimbType>& left, const uintwide_t<Digits2, LimbType>& right) { return uintwide_t<Digits2, LimbType>(left).operator&=(right); }

  // Non-member binary add, sub, mul, div, mod of (uintwide_t op IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator+(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator+=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator-(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator-=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator*(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator*=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator/(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator/=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator%(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator%=(uintwide_t<Digits2, LimbType>(v)); }

  // Non-member binary add, sub, mul, div, mod of (IntegralType op uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator+(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator+=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator-(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator-=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator*(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator*=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator/(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator/=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator%(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator%=(v); }

  // Non-member binary logic operations of (uintwide_t op IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator|(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator|=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator^(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator^=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator&(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return uintwide_t<Digits2, LimbType>(u).operator&=(uintwide_t<Digits2, LimbType>(v)); }

  // Non-member binary binary logic operations of (IntegralType op uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator|(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator|=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator^(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator^=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator&(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator&=(v); }

  // Non-member shift functions of (uintwide_t shift IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator<<(const uintwide_t<Digits2, LimbType>& u, const IntegralType n) { return uintwide_t<Digits2, LimbType>(u).operator<<=(n); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), uintwide_t<Digits2, LimbType>>::type
  operator>>(const uintwide_t<Digits2, LimbType>& u, const IntegralType n) { return uintwide_t<Digits2, LimbType>(u).operator>>=(n); }

  // Non-member comparison functions of (uintwide_t cmp uintwide_t).
  template<const std::size_t Digits2, typename LimbType> bool operator==(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator==(v); }
  template<const std::size_t Digits2, typename LimbType> bool operator!=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator!=(v); }
  template<const std::size_t Digits2, typename LimbType> bool operator> (const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator> (v); }
  template<const std::size_t Digits2, typename LimbType> bool operator< (const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator< (v); }
  template<const std::size_t Digits2, typename LimbType> bool operator>=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator>=(v); }
  template<const std::size_t Digits2, typename LimbType> bool operator<=(const uintwide_t<Digits2, LimbType>& u, const uintwide_t<Digits2, LimbType>& v) { return u.operator<=(v); }

  // Non-member comparison functions of (uintwide_t cmp IntegralType).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator==(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator==(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator!=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator!=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator> (const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator> (uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator< (const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator< (uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator>=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator>=(uintwide_t<Digits2, LimbType>(v)); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator<=(const uintwide_t<Digits2, LimbType>& u, const IntegralType& v) { return u.operator<=(uintwide_t<Digits2, LimbType>(v)); }

  // Non-member comparison functions of (IntegralType cmp uintwide_t).
  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator==(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator==(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator!=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator!=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator> (const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator> (v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator< (const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator< (v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator>=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator>=(v); }

  template<typename IntegralType, const std::size_t Digits2, typename LimbType>
  typename std::enable_if<(   (std::is_fundamental<IntegralType>::value == true)
                           && (std::is_integral   <IntegralType>::value == true)), bool>::type
  operator<=(const IntegralType& u, const uintwide_t<Digits2, LimbType>& v) { return uintwide_t<Digits2, LimbType>(u).operator<=(v); }

  #if defined(WIDE_INTEGER_DISABLE_IOSTREAM)
  #else

  // I/O streaming functions.
  template<typename char_type,
           typename traits_type,
           const std::size_t Digits2,
           typename LimbType>
  std::basic_ostream<char_type, traits_type>&
  operator<<(std::basic_ostream<char_type, traits_type>& out,
             const uintwide_t<Digits2, LimbType>& x)
  {
    std::basic_ostringstream<char_type, traits_type> ostr;

    const std::ios::fmtflags my_flags = out.flags();

    const bool show_pos     = ((my_flags & std::ios::showpos)   == std::ios::showpos);
    const bool show_base    = ((my_flags & std::ios::showbase)  == std::ios::showbase);
    const bool is_uppercase = ((my_flags & std::ios::uppercase) == std::ios::uppercase);

    std::uint_fast8_t base_rep;

    if     ((my_flags & std::ios::oct) == std::ios::oct) { base_rep =  8U; }
    else if((my_flags & std::ios::hex) == std::ios::hex) { base_rep = 16U; }
    else                                                 { base_rep = 10U; }

    const std::size_t field_width = std::size_t(out.width());
    const char        fill_char   = out.fill();

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;

    if(base_rep == 8U)
    {
      char str_result[local_wide_integer_type::wr_string_max_buffer_size_oct];

      x.wr_string(str_result, base_rep, show_base, show_pos, is_uppercase, field_width, fill_char);

      static_cast<void>(ostr << str_result);
    }
    else if(base_rep == 10U)
    {
      char str_result[local_wide_integer_type::wr_string_max_buffer_size_dec];

      x.wr_string(str_result, base_rep, show_base, show_pos, is_uppercase, field_width, fill_char);

      static_cast<void>(ostr << str_result);
    }
    else if(base_rep == 16U)
    {
      char str_result[local_wide_integer_type::wr_string_max_buffer_size_hex];

      x.wr_string(str_result, base_rep, show_base, show_pos, is_uppercase, field_width, fill_char);

      static_cast<void>(ostr << str_result);
    }

    return (out << ostr.str());
  }

  template<typename char_type,
           typename traits_type,
           const std::size_t Digits2,
           typename LimbType>
  std::basic_istream<char_type, traits_type>&
  operator>>(std::basic_istream<char_type, traits_type>& in,
             uintwide_t<Digits2, LimbType>& x)
  {
    std::string str_in;

    in >> str_in;

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;

    x = local_wide_integer_type(str_in.c_str());

    return in;
  }

  #endif

  } } // namespace wide_integer::generic_template

  // Implement various number-theoretical tools.

  namespace wide_integer { namespace generic_template {

  template<const std::size_t Digits2,
           typename LimbType>
  void swap(uintwide_t<Digits2, LimbType>& x,
            uintwide_t<Digits2, LimbType>& y)
  {
    if(&x != &y)
    {
      using local_wide_integer_type = uintwide_t<Digits2, LimbType>;

      const local_wide_integer_type tmp_x(x);

      x = y;
      y = tmp_x;
    }
  }

  template<const std::size_t Digits2,
           typename LimbType>
  std::size_t lsb(const uintwide_t<Digits2, LimbType>& x)
  {
    // Calculate the position of the least-significant bit.

    using local_wide_integer_type   = uintwide_t<Digits2, LimbType>;
    using local_const_iterator_type = typename local_wide_integer_type::const_iterator;
    using local_value_type          = typename local_wide_integer_type::value_type;

    std::size_t bpos = 0U;

    for(local_const_iterator_type it = x.crepresentation().cbegin(); it != x.crepresentation().cend(); ++it)
    {
      if((*it & (std::numeric_limits<local_value_type>::max)()) != 0U)
      {
        const std::size_t offset = std::size_t(it - x.crepresentation().cbegin());

        bpos =   detail::lsb_helper(*it)
               + std::size_t(std::size_t(std::numeric_limits<local_value_type>::digits) * offset);

        break;
      }
    }

    return bpos;
  }

  template<const std::size_t Digits2,
           typename LimbType>
  std::size_t msb(const uintwide_t<Digits2, LimbType>& x)
  {
    // Calculate the position of the most-significant bit.

    using local_wide_integer_type           = uintwide_t<Digits2, LimbType>;
    using local_const_reverse_iterator_type = typename local_wide_integer_type::const_reverse_iterator;
    using local_value_type                  = typename local_wide_integer_type::value_type;

    std::size_t bpos = 0U;

    for(local_const_reverse_iterator_type ri = x.crepresentation().crbegin(); ri != x.crepresentation().crend(); ++ri)
    {
      if((*ri & (std::numeric_limits<local_value_type>::max)()) != 0U)
      {
        const std::size_t offset = std::size_t((x.crepresentation().crend() - 1U) - ri);

        bpos =   detail::msb_helper(*ri)
               + std::size_t(std::size_t(std::numeric_limits<local_value_type>::digits) * offset);

        break;
      }
    }

    return bpos;
  }

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> sqrt(const uintwide_t<Digits2, LimbType>& m)
  {
    // Calculate the square root.

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;
    using local_value_type        = typename local_wide_integer_type::value_type;

    const bool argument_is_zero = std::all_of(m.crepresentation().cbegin(),
                                              m.crepresentation().cend(),
                                              [](const local_value_type& a) -> bool
                                              {
                                                return (a == 0U);
                                              });

    local_wide_integer_type s;

    if(argument_is_zero)
    {
      s = local_wide_integer_type(std::uint_fast8_t(0U));
    }
    else
    {
      // Obtain the initial guess via algorithms
      // involving the position of the msb.
      const std::size_t msb_pos = msb(m);

      // Obtain the initial value.
      const std::size_t left_shift_amount =
        ((std::size_t(msb_pos % 2U) == 0U) ? 1U + std::size_t((msb_pos + 0U) / 2U)
                                           : 1U + std::size_t((msb_pos + 1U) / 2U));

      local_wide_integer_type u(local_wide_integer_type(std::uint_fast8_t(1U)) << left_shift_amount);

      // Perform the iteration for the square root.
      // See Algorithm 1.13 SqrtInt, Sect. 1.5.1
      // in R.P. Brent and Paul Zimmermann, "Modern Computer Arithmetic",
      // Cambridge University Press, 2011.

      for(std::size_t i = 0U; i < 64U; ++i)
      {
        s = u;

        u = (s + (m / s)) >> 1;

        if(u >= s)
        {
          break;
        }
      }
    }

    return s;
  }

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> cbrt(const uintwide_t<Digits2, LimbType>& m)
  {
    return rootk(m, 3U);
  }

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> rootk(const uintwide_t<Digits2, LimbType>& m,
                                      const std::uint_fast8_t k)
  {
    // Calculate the k'th root.

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;
    using local_value_type        = typename local_wide_integer_type::value_type;

    local_wide_integer_type s;

    if(k < 2U)
    {
      s = m;
    }
    else if(k == 2U)
    {
      s = sqrt(m);
    }
    else
    {
      const bool argument_is_zero = std::all_of(m.crepresentation().cbegin(),
                                                m.crepresentation().cend(),
                                                [](const local_value_type& a) -> bool
                                                {
                                                  return (a == 0U);
                                                });

      if(argument_is_zero)
      {
        s = local_wide_integer_type(std::uint_fast8_t(0U));
      }
      else
      {
        // Obtain the initial guess via algorithms
        // involving the position of the msb.
        const std::size_t msb_pos = msb(m);

        // Obtain the initial value.
        const std::size_t msb_pos_mod_k = msb_pos % k;

        const std::size_t left_shift_amount =
          ((msb_pos_mod_k == 0U) ? 1U + std::size_t((msb_pos +                 0U ) / k)
                                 : 1U + std::size_t((msb_pos + (k - msb_pos_mod_k)) / k));

        local_wide_integer_type u(local_wide_integer_type(std::uint_fast8_t(1U)) << left_shift_amount);

        // Perform the iteration for the k'th root.
        // See Algorithm 1.14 RootInt, Sect. 1.5.2
        // in R.P. Brent and Paul Zimmermann, "Modern Computer Arithmetic",
        // Cambridge University Press, 2011.

        const local_wide_integer_type k_minus_one(k - 1U);

        for(std::size_t i = 0U; i < 64U; ++i)
        {
          s = u;

          local_wide_integer_type m_over_s_pow_k_minus_one = m;

          for(std::size_t j = 0U; j < k - 1U; ++j)
          {
            // Use a loop here to divide by s^(k - 1) because
            // without a loop, s^(k - 1) is likely to overflow.

            m_over_s_pow_k_minus_one /= s;
          }

          u = ((s * k_minus_one) + m_over_s_pow_k_minus_one) / k;

          if(u >= s)
          {
            break;
          }
        }
      }
    }

    return s;
  }

  template<typename OtherUnsignedIntegralTypeP,
           const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> pow(const uintwide_t<Digits2, LimbType>& b,
                                    const OtherUnsignedIntegralTypeP&    p)
  {
    // Calculate (b ^ p).

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;

    local_wide_integer_type result;

    const OtherUnsignedIntegralTypeP zero(std::uint8_t(0U));

    if(p == zero)
    {
      result = local_wide_integer_type(std::uint8_t(1U));
    }
    else if(p == 1U)
    {
      result = b;
    }
    else if(p == 2U)
    {
      result  = b;
      result *= b;
    }
    else
    {
      result = local_wide_integer_type(std::uint8_t(1U));

      local_wide_integer_type y      (b);
      local_wide_integer_type p_local(p);

      while(!(p_local == zero))
      {
        if(std::uint_fast8_t(p_local) & 1U)
        {
          result *= y;
        }

        y *= y;

        p_local >>= 1;
      }
    }

    return result;
  }

  template<typename OtherUnsignedIntegralTypeP,
           typename OtherUnsignedIntegralTypeM,
           const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> powm(const uintwide_t<Digits2, LimbType>& b,
                                     const OtherUnsignedIntegralTypeP&    p,
                                     const OtherUnsignedIntegralTypeM&    m)
  {
    // Calculate (b ^ p) % m.

    using local_normal_width_type = uintwide_t<Digits2, LimbType>;
    using local_double_width_type = typename local_normal_width_type::double_width_type;

          local_normal_width_type    result;
    const OtherUnsignedIntegralTypeP zero   (std::uint8_t(0U));
          local_double_width_type    y      (b);
    const local_double_width_type    m_local(m);

    if(p == zero)
    {
      result = local_normal_width_type((m != 1U) ? std::uint8_t(1U) : std::uint8_t(0U));
    }
    else if(p == 1U)
    {
      result = b % m;
    }
    else if(p == 2U)
    {
      y *= y;

      result = local_normal_width_type(y %= m_local);
    }
    else
    {
      local_double_width_type    x      (std::uint8_t(1U));
      OtherUnsignedIntegralTypeP p_local(p);

      while(!(p_local == zero))
      {
        if(std::uint_fast8_t(p_local) & 1U)
        {
          x *= y;
          x %= m_local;
        }

        y *= y;
        y %= m_local;

        p_local >>= 1;
      }

      result = local_normal_width_type(x);
    }

    return result;
  }

  template<const std::size_t Digits2,
           typename LimbType>
  uintwide_t<Digits2, LimbType> gcd(const uintwide_t<Digits2, LimbType>& a,
                                    const uintwide_t<Digits2, LimbType>& b)
  {
    // This implementation of GCD is an adaptation
    // of existing code from Boost.Multiprecision.

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;
    using local_ushort_type       = typename local_wide_integer_type::ushort_type;
    using local_ularge_type       = typename local_wide_integer_type::ularge_type;

    local_wide_integer_type u(a);
    local_wide_integer_type v(b);

    local_wide_integer_type result;

    if(u == v)
    {
      // This handles cases having (u = v) and also (u = v = 0).
      result = u;
    }
    else if(v == 0U)
    {
      // This handles cases having (v = 0) with (u != 0).
      result = u;
    }
    else if(u == 0U)
    {
      // This handles cases having (u = 0) with (v != 0).
      result = v;
    }
    else
    {
      // Now we handle cases having (u != 0) and (v != 0).

      // Let shift := lg K, where K is the greatest
      // power of 2 dividing both u and v.

      std::size_t left_shift_amount;

      {
        const std::size_t u_shift = lsb(u);
        const std::size_t v_shift = lsb(v);

        left_shift_amount = (std::min)(u_shift, v_shift);

        u >>= u_shift;
        v >>= v_shift;
      }

      for(;;)
      {
        // Now u and v are both odd, so diff(u, v) is even.
        // Let u = min(u, v), v = diff(u, v) / 2.

        if(u > v)
        {
          swap(u, v);
        }

        if(u == v)
        {
          break;
        }

        if(v <= (std::numeric_limits<local_ularge_type>::max)())
        {
          if(v <= (std::numeric_limits<local_ushort_type>::max)())
          {
            u = detail::integer_gcd_reduce_short(v.crepresentation()[0U],
                                                 u.crepresentation()[0U]);
          }
          else
          {
            const local_ularge_type v_large =
              detail::make_large(v.crepresentation()[0U],
                                 v.crepresentation()[1U]);

            const local_ularge_type u_large =
              detail::make_large(u.crepresentation()[0U],
                                 u.crepresentation()[1U]);

            u = detail::integer_gcd_reduce_large(v_large, u_large);
          }

          break;
        }

        v  -= u;
        v >>= lsb(v);
      }

      result = (u << left_shift_amount);
    }

    return result;
  }

  template<const std::size_t Digits2,
           typename LimbType>
  class default_random_engine
  {
  public:
    // Use a fast and efficient PCG-family random number generator.

    using result_type = uintwide_t<Digits2, LimbType>;
    using value_type  = std::uint32_t;

    static const value_type default_seed = 0U;

    explicit default_random_engine(const value_type new_seed = default_seed)
      : my_state(0U),
        my_inc  (0U)
    {
      seed(new_seed);
    }

    void seed(const value_type new_seed = default_seed)
    {
      const std::uint64_t initstate = crc64_we(new_seed);
      const std::uint64_t initseq   = 0U;

      my_inc = std::uint64_t(initseq << 1) | 1U;
      step();
      my_state += initstate;
      step();
    }

    result_type operator()()
    {
      result_type result(std::uint_fast8_t(0U));

      using local_result_value_type = typename result_type::value_type;

      const std::size_t digits_ratio = 
        std::size_t(  std::numeric_limits<local_result_value_type>::digits
                    / std::numeric_limits<value_type>::digits);

      switch(digits_ratio)
      {
        case 0:
          // The limbs in the wide integer result are less wide than
          // the 32-bit width of the random number generator result.
          {
            const std::size_t digits_ratio_inverse = 
              std::size_t(  std::numeric_limits<value_type>::digits
                          / std::numeric_limits<local_result_value_type>::digits);

            auto it = result.representation().begin();

            while(it < result.representation().end())
            {
              const value_type value = next_random_value();

              for(std::size_t j = 0U; j < digits_ratio_inverse; ++j)
              {
                *(it + j) |= local_result_value_type(value >> (j * std::size_t(std::numeric_limits<local_result_value_type>::digits)));
              }

              it += digits_ratio_inverse;
            }
          }
          break;

        case 1:
          // The limbs in the wide integer result are equally as wide as
          // the 32-bit width of the random number generator result.
          for(auto it = result.representation().begin(); it != result.representation().end(); ++it)
          {
            *it = next_random_value();
          }
          break;

        default:
          // The limbs in the wide integer result are wider than
          // the 32-bit width of the random number generator result.
          for(auto it = result.representation().begin(); it != result.representation().end(); ++it)
          {
            for(std::size_t j = 0U; j < digits_ratio; ++j)
            {
              const local_result_value_type value = local_result_value_type(next_random_value());

              const std::size_t left_shift_amount =
                std::size_t(j * std::size_t(std::numeric_limits<value_type>::digits));

              (*it) |= local_result_value_type(value << left_shift_amount);
            }
          }
          break;
      }

      return result;
    }

  private:
    std::uint64_t my_state;
    std::uint64_t my_inc;

    static const std::uint64_t default_multiplier = UINT64_C(6364136223846793005);

    value_type next_random_value()
    {
      const std::uint64_t previous_state = my_state;

      step();

      const value_type next_value =
        rotate(value_type      (((previous_state >> 18U) ^ previous_state) >> 27U),
               std::int_fast8_t  (previous_state >> 59U));

      return next_value;
    }

    void step()
    {
      my_state = std::uint64_t(std::uint64_t(my_state * default_multiplier) + my_inc);
    }

    static value_type rotate(value_type value, std::int_fast8_t rot)
    {
      return value_type(value_type(value >> rot)
                                | (value << std::int_fast8_t(std::uint_fast8_t(-rot) & 31U)));
    }

    template<typename UnsignedIntegralType>
    std::uint64_t crc64_we(const UnsignedIntegralType v)
    {
      // Calculate a bitwise CRC64/WE over the
      // individual bytes of the input parameter v.

      // Extract the bytes of v into an array.
      std::array<std::uint8_t, std::numeric_limits<UnsignedIntegralType>::digits / 8U> data;

      for(std::uint_fast8_t i = 0U; i < data.size(); ++ i)
      {
        data[i] = std::uint8_t(v >> (i * 8U));
      }

      std::uint64_t crc = UINT64_C(0xFFFFFFFFFFFFFFFF);

      // Perform modulo-2 division, one byte at a time.
      for(std::size_t byte = 0U; byte < data.size(); ++byte)
      {
        // Bring the next byte into the result.
        crc ^= (std::uint64_t(data[byte]) << (std::numeric_limits<std::uint64_t>::digits - 8U));

        // Perform a modulo-2 division, one bit at a time.
        for(std::int_fast8_t bit = 8; bit > 0; --bit)
        {
          // Divide the current data bit.
          if((crc & (std::uintmax_t(1U) << (std::numeric_limits<std::uint64_t>::digits - 1U))) != 0U)
          {
            crc = std::uint64_t(crc << 1) ^ UINT64_C(0x42F0E1EBA9EA3693);
          }
          else
          {
            crc <<= 1;
          }
        }
      }

      return std::uint64_t(crc ^ UINT64_C(0xFFFFFFFFFFFFFFFF));
    }
  };

  template<const std::size_t Digits2,
           typename LimbType>
  class uniform_int_distribution
  {
  public:
    using result_type = uintwide_t<Digits2, LimbType>;

    struct param_type
    {
    public:
      param_type(const result_type& a = (std::numeric_limits<result_type>::min)(),
                 const result_type& b = (std::numeric_limits<result_type>::max)())
        : param_a(a),
          param_b(b) { }

      ~param_type() = default;

      param_type(const param_type& other_params) : param_a(other_params.param_a),
                                                   param_b(other_params.param_b) { }

      param_type& operator=(const param_type& other_params)
      {
        if(this != &other_params)
        {
          param_a = other_params.param_a;
          param_b = other_params.param_b;
        }

        return *this;
      }

      const result_type& get_a() const { return param_a; }
      const result_type& get_b() const { return param_b; }

      void set_a(const result_type& a) { param_a = a; }
      void set_b(const result_type& b) { param_b = b; }

    private:
      result_type param_a;
      result_type param_b;
    };

    uniform_int_distribution() : my_params() { }

    explicit uniform_int_distribution(const result_type& a,
                                      const result_type& b = (std::numeric_limits<result_type>::max)())
        : my_params(param_type(a, b)) { }

    explicit uniform_int_distribution(const param_type& other_params)
      : my_params(other_params) { }

    uniform_int_distribution(const uniform_int_distribution& other_distribution)
      : my_params(other_distribution.my_params) { }

    ~uniform_int_distribution() = default;

    uniform_int_distribution& operator=(const uniform_int_distribution& other_distribution)
    {
      if(this != &other_distribution)
      {
        my_params = other_distribution.my_params;
      }

      return *this;
    }

    void params(const param_type& new_params)
    {
      my_params = new_params;
    }

    const param_type& params() const { return my_params; }

    template<typename GeneratorType>
    result_type operator()(GeneratorType& generator)
    {
      return generate(generator, my_params);
    }

    template<typename GeneratorType>
    result_type operator()(GeneratorType& input_generator,
                           const param_type& input_params)
    {
      return generate(input_generator, input_params);
    }

  private:
    param_type my_params;

    template<typename GeneratorType>
    result_type generate(GeneratorType& input_generator,
                         const param_type& input_params)
    {
      result_type result = input_generator();

      if(   (input_params.get_a() != (std::numeric_limits<result_type>::min)())
         || (input_params.get_b() != (std::numeric_limits<result_type>::max)()))
      {
        result_type range(input_params.get_b() - input_params.get_a());
        ++range;

        result %= range;
        result += input_params.get_a();
      }

      return result;
    }
  };

  template<const std::size_t Digits2,
           typename LimbType>
  bool operator==(const uniform_int_distribution<Digits2, LimbType>& lhs,
                  const uniform_int_distribution<Digits2, LimbType>& rhs)
  {
    return (   (lhs.params().get_a() == rhs.params().get_a())
            && (lhs.params().get_b() == rhs.params().get_b()));
  }

  template<const std::size_t Digits2,
           typename LimbType>
  bool operator!=(const uniform_int_distribution<Digits2, LimbType>& lhs,
                  const uniform_int_distribution<Digits2, LimbType>& rhs)
  {
    return (   (lhs.params().get_a() != rhs.params().get_a())
            || (lhs.params().get_b() != rhs.params().get_b()));
  }

  template<typename DistributionType,
           typename GeneratorType,
           const std::size_t Digits2,
           typename LimbType>
  bool miller_rabin(const uintwide_t<Digits2, LimbType>& n,
                    const std::size_t                    number_of_trials,
                    DistributionType&                    distribution,
                    GeneratorType&                       generator)
  {
    // This Miller-Rabin primality test is loosely based on
    // an adaptation of some code from Boost.Multiprecision.
    // The Boost.Multiprecision code can be found here:
    // https://www.boost.org/doc/libs/1_68_0/libs/multiprecision/doc/html/boost_multiprecision/tut/primetest.html

    // Note: Some comments in this subroutine use the Wolfram Language(TM).

    using local_wide_integer_type = uintwide_t<Digits2, LimbType>;

    const std::uint_fast8_t n8(n);

    if((n8 == 2U) && (n == 2U))
    {
      // Trivial special case of (n = 2).
      return true;
    }

    if((n8 & 1U) == 0U)
    {
      // Not prime because n is even.
      return false;
    }

    if((n8 <= 227U) && (n <= 227U))
    {
      // Table[Prime[i], {i, 2, 49}] =
      // {
      //     3,   5,   7,  11,  13,  17,  19,  23,
      //    29,  31,  37,  41,  43,  47,  53,  59,
      //    61,  67,  71,  73,  79,  83,  89,  97,
      //   101, 103, 107, 109, 113, 127, 131, 137,
      //   139, 149, 151, 157, 163, 167, 173, 179,
      //   181, 191, 193, 197, 199, 211, 223, 227
      // }

      // Exclude pure small primes from 3...227.
      constexpr std::array<std::uint_fast8_t, 48U> small_primes = 
      {{
        UINT8_C(  3), UINT8_C(  5), UINT8_C(  7), UINT8_C( 11), UINT8_C( 13), UINT8_C( 17), UINT8_C( 19), UINT8_C( 23),
        UINT8_C( 29), UINT8_C( 31), UINT8_C( 37), UINT8_C( 41), UINT8_C( 43), UINT8_C( 47), UINT8_C( 53), UINT8_C( 59),
        UINT8_C( 61), UINT8_C( 67), UINT8_C( 71), UINT8_C( 73), UINT8_C( 79), UINT8_C( 83), UINT8_C( 89), UINT8_C( 97),
        UINT8_C(101), UINT8_C(103), UINT8_C(107), UINT8_C(109), UINT8_C(113), UINT8_C(127), UINT8_C(131), UINT8_C(137),
        UINT8_C(139), UINT8_C(149), UINT8_C(151), UINT8_C(157), UINT8_C(163), UINT8_C(167), UINT8_C(173), UINT8_C(179),
        UINT8_C(181), UINT8_C(191), UINT8_C(193), UINT8_C(197), UINT8_C(199), UINT8_C(211), UINT8_C(223), UINT8_C(227)
      }};

      return std::binary_search(small_primes.cbegin(),
                                small_primes.cend(),
                                n8);
    }

    // Check small factors.
    {
      // Product[Prime[i], {i, 2, 16}] = 16294579238595022365
      // Exclude small prime factors from { 3 ...  53 }.
      constexpr std::uint64_t pp0 = UINT64_C(16294579238595022365);

      const std::uint64_t m0(n % pp0);

      if(detail::integer_gcd_reduce_large(m0, pp0) != 1U)
      {
        return false;
      }
    }

    {
      // Product[Prime[i], {i, 17, 26}] = 7145393598349078859
      // Exclude small prime factors from { 59 ... 101 }.
      constexpr std::uint64_t pp1 = UINT64_C(7145393598349078859);

      const std::uint64_t m1(n % pp1);

      if(detail::integer_gcd_reduce_large(m1, pp1) != 1U)
      {
        return false;
      }
    }

    {
      // Product[Prime[i], {i, 27, 35}] = 6408001374760705163
      // Exclude small prime factors from { 103 ... 149 }.
      constexpr std::uint64_t pp2 = UINT64_C(6408001374760705163);

      const std::uint64_t m2(n % pp2);

      if(detail::integer_gcd_reduce_large(m2, pp2) != 1U)
      {
        return false;
      }
    }

    {
      // Product[Prime[i], {i, 36, 43}] = 690862709424854779
      // Exclude small prime factors from { 151 ... 191 }.
      constexpr std::uint64_t pp3 = UINT64_C(690862709424854779);

      const std::uint64_t m3(n % pp3);

      if(detail::integer_gcd_reduce_large(m3, pp3) != 1U)
      {
        return false;
      }
    }

    {
      // Product[Prime[i], {i, 44, 49}] = 80814592450549
      // Exclude small prime factors from { 193 ... 227 }.
      constexpr std::uint64_t pp4 = UINT64_C(80814592450549);

      const std::uint64_t m4(n % pp4);

      if(detail::integer_gcd_reduce_large(m4, pp4) != 1U)
      {
        return false;
      }
    }

    const local_wide_integer_type nm1(n - 1U);

    // Perform a single Fermat test which will
    // exclude many non-prime candidates.

    // We know now that n is greater than 227 because
    // we have already excluded all small factors
    // up to and including 227.
    local_wide_integer_type q(std::uint_fast8_t(228U));

    if(powm(q, nm1, n) != 1U)
    {
      return false;
    }

    const std::size_t k = lsb(nm1);
    q = nm1 >> k;

    const typename DistributionType::param_type params(local_wide_integer_type(2U),
                                                       local_wide_integer_type(n - 2U));

    // Execute the random trials.
    for(std::size_t i = 0U; i < number_of_trials; ++i)
    {
      local_wide_integer_type x = distribution(generator, params);
      local_wide_integer_type y = powm(x, q, n);

      std::size_t j = 0U;

      // TBD: The following code seems convoluded and it is difficult
      // to understand this code. Can this while-loop and all returns
      // and breaks be written in a more intuitive and clear form?
      while(true)
      {
        if(y == nm1)
        {
          break;
        }

        if(y == 1U)
        {
          if(j == 0U)
          {
            break;
          }

          return false;
        }

        ++j;

        if(j == k)
        {
          return false;
        }

        y = powm(y, 2U, n);
      }
    }

    // Probably prime.
    return true;
  }

  } } // namespace wide_integer::generic_template

#endif // GENERIC_TEMPLATE_UINTWIDE_T_2018_10_02_H_
