#ifndef PILOT_EXCEPTION_H
#define PILOT_EXCEPTION_H

#include "QPandaConfig.h"
#include <string>
#include <exception>
#include <stdexcept>
#include "Extensions/PilotOSMachine/OSDef.h"

/**
 * @class PilotException
 * @brief Pilot exception basic class
 */
class PilotException : public std::runtime_error
{
public:
    PilotException() : runtime_error(m_err_msg.c_str()) {}

    PilotException(PilotQVM::ErrorCode err_code, const std::string& err_str) : runtime_error(err_str.c_str())
    {
        m_err_msg = err_str;
        m_err_code = err_code;
    }

    std::string get_err_msg() const { return m_err_msg; }
    PilotQVM::ErrorCode get_err_code() const { return m_err_code; }

private:
    std::string m_err_msg{ "PilotErr: Unknown error." };
    PilotQVM::ErrorCode m_err_code{ PilotQVM::ErrorCode::NO_ERROR_FOUND };
};

/*
 * @param[in]  con judging condition
 * @param[in]  err_code PilotQVM::ErrorCode
 * @param[in]  err_msg string
 */
#define PILOT_ASSERT(con, err_code, err_msg) {if(con){throw PilotException(err_code, err_msg);}}

#endif // !PILOT_EXCEPTION_H
