#ifndef _QMATRIXDEF_H_
#define _QMATRIXDEF_H_
#include "Core/Utilities/QPandaNamespace.h"
#include "ThirdParty/Eigen/Eigen"

QPANDA_BEGIN

/** \defgroup OriginQ global matrix typedefs
*/
#define Q_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
typedef Eigen::Matrix<Type, Size, Size, Eigen::RowMajor> QMatrix##SizeSuffix##TypeSuffix;  \
typedef Eigen::Matrix<Type, Size, 1, Eigen::ColMajor> QVector##SizeSuffix##TypeSuffix;  \
typedef Eigen::Matrix<Type, 1, Size, Eigen::ColMajor> QRowVector##SizeSuffix##TypeSuffix;

#define Q_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
typedef Eigen::Matrix<Type, Size, Eigen::Dynamic, Eigen::RowMajor> QMatrix##Size##X##TypeSuffix;  \
typedef Eigen::Matrix<Type, Eigen::Dynamic, Size,  Eigen::RowMajor> QMatrix##X##Size##TypeSuffix;


#define Q_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
Q_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
Q_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
Q_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
Q_MAKE_TYPEDEFS(Type, TypeSuffix, Eigen::Dynamic, X) \
Q_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
Q_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
Q_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

Q_MAKE_TYPEDEFS_ALL_SIZES(int, i)
Q_MAKE_TYPEDEFS_ALL_SIZES(float, f)
Q_MAKE_TYPEDEFS_ALL_SIZES(double, d)
Q_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>, cf)
Q_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

#undef Q_MAKE_TYPEDEFS_ALL_SIZES
#undef Q_MAKE_TYPEDEFS
#undef Q_MAKE_FIXED_TYPEDEFS

QPANDA_END
#endif //!_QMATRIXDEF_H_