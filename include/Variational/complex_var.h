#ifndef COMPLEX_VAR_H
#define COMPLEX_VAR_H

#include "var.h"

QPANDA_BEGIN

using namespace Variational;

class complex_var
{
public:
	complex_var():
		m_real(0),
		m_imag(0)
	{
	}

	complex_var(const var &value) :
		m_real(value),
		m_imag(0)
	{}
	
	complex_var(const var &v1, const var &v2):
		m_real(v1),
		m_imag(v2)
	{}

    var real()
    {
        return m_real;
    }

    var imag()
    {
        return m_imag;
    }

	const complex_var operator + (const complex_var &value)
	{
		return complex_var(m_real + value.m_real, m_imag + value.m_imag);
	}

    const complex_var operator - (const complex_var &value)
    {
        return complex_var(m_real - value.m_real, m_imag - value.m_imag);
    }

    // (a+bi)*(c+di)=(ac-bd)+(ad+bc)i
    const complex_var operator * (const complex_var &value)
    {
        return complex_var(m_real*value.m_real - m_imag*value.m_imag,
            m_real*value.m_imag + m_imag*value.m_real);
    }

    // (a+bi)/(c+di) = (ac+bd)/(c*c+d*d)+(bc-ad)i/(c*c+d*d)
    const complex_var operator / (const complex_var &value)
    {
        return complex_var((m_real*value.m_real+m_imag*value.m_imag)/((value.m_real,2)+poly(value.m_imag,2))
            , (m_imag*value.m_real - m_real*value.m_imag)/ (poly(value.m_real, 2) + poly(value.m_imag, 2)));
    }

    friend const complex_var operator + (const complex_var& lhs, const complex_var& rhs)
    {
        return complex_var(lhs.m_real + rhs.m_real, lhs.m_imag + rhs.m_imag);
    }

    friend const complex_var operator - (const complex_var& lhs, const complex_var& rhs)
    {
        return complex_var(lhs.m_real - rhs.m_real, lhs.m_imag - rhs.m_imag);
    }

    // (a+bi)*(c+di)=(ac-bd)+(ad+bc)i
    friend const complex_var operator * (const complex_var& lhs, const complex_var& rhs)
    {
        return complex_var(lhs.m_real * rhs.m_real - lhs.m_imag * rhs.m_imag,
            lhs.m_real * rhs.m_imag + lhs.m_imag * rhs.m_real);
    }

    // (a+bi)/(c+di) = (ac+bd)/(c*c+d*d)+(bc-ad)i/(c*c+d*d)
    friend const complex_var operator / (const complex_var& lhs, const complex_var& rhs)
    {
        return complex_var((lhs.m_real * rhs.m_real + lhs.m_imag * rhs.m_imag) / (poly(rhs.m_real, 2) + poly(rhs.m_imag, 2))
            , (lhs.m_imag * rhs.m_real - lhs.m_real * rhs.m_imag) / (poly(rhs.m_real, 2) + poly(rhs.m_imag, 2)));
    }

private:
	var m_real{0};
	var m_imag{0};
};

QPANDA_END

#endif//COMPLEX_VAR_H