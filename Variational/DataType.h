#ifndef DATATYPE_H
#define DATATYPE_H

#include <memory>

namespace QPanda {
namespace Variational {

class Double
{
    std::shared_ptr<double> m_d;
public:
    Double();
    Double(double);
    Double(const Double&);
    operator double()
    {
        return *m_d;
    }
};

inline Double::Double() { m_d = std::shared_ptr<double>(new double(0)); }
inline Double::Double(double val) { m_d = std::shared_ptr<double>(new double(val)); }
inline Double::Double(const Double& d) { m_d = d.m_d; }

} // Variational
} // QPanda
#endif // !DATATYPE_H