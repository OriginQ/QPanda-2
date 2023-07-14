#pragma once

#include <string>
#include <atomic>
#include <system_error>
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

enum class LogLevel
{
    CLOUD_INFO,
    CLOUD_DEBUG,
    CLOUD_WARNING,
    CLOUD_ERROR
};

class QCloudLogger 
{

public:

    static QCloudLogger& get_instance() 
    {
        static QCloudLogger instance;
        return instance;
    }

    void enable() 
    {
        m_enabled = true;
    }

    void disable() 
    {
        m_enabled = false;
    }

    bool is_enabled() const 
    {
        return m_enabled;
    }

private:
    std::atomic<bool> m_enabled;

    QCloudLogger() : m_enabled(false) {}
    QCloudLogger(const QCloudLogger&) = delete;
    QCloudLogger& operator=(const QCloudLogger&) = delete;

};

void qcloud_log_out(LogLevel level, const std::string& message, const std::string& file, int line);

#define LOG_WARNING(message) qcloud_log_out(LogLevel::CLOUD_WARNING, message, __FILENAME__, __LINE__)
#define LOG_DEBUG(message)   qcloud_log_out(LogLevel::CLOUD_DEBUG,   message, __FILENAME__, __LINE__)
#define LOG_INFO(message)    qcloud_log_out(LogLevel::CLOUD_INFO,    message, __FILENAME__, __LINE__)
#define LOG_ERROR(message)   qcloud_log_out(LogLevel::CLOUD_ERROR,   message, __FILENAME__, __LINE__)

QPANDA_END
