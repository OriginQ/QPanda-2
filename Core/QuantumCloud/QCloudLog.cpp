#include <iostream>
#include "Core/QuantumCloud/QCloudLog.h"

USING_QPANDA

void QPanda::qcloud_log_out(LogLevel level, const std::string& message, const std::string& file, int line)
{
    auto& logger = QCloudLogger::get_instance();

    if (logger.is_enabled()) 
    {
        std::string message_out;
        switch (level)
        {
        case LogLevel::CLOUD_DEBUG:   message_out = ("DEBUG : "   + message); break;
        case LogLevel::CLOUD_INFO:    message_out = ("INFO : "    + message); break;
        case LogLevel::CLOUD_WARNING: message_out = ("WARNING : " + message); break;
        case LogLevel::CLOUD_ERROR:   message_out = ("ERROR : "   + message); break;
        default:  break;
        }

        std::cerr << file << ":" << line << " " << message_out << std::endl;
    }
}