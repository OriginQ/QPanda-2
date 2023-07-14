#pragma once

#include <string>
#include <system_error>
#include "Core/Utilities/QPandaNamespace.h"

#include "QPandaConfig.h"
#if defined(USE_OPENSSL) && defined(USE_CURL)

#include <curl/curl.h>
using pcurl_t = CURL * ;
using pcurl_slist_t = curl_slist * ;

QPANDA_BEGIN

class QCurl
{
private:

    pcurl_t  m_curl_handle = nullptr;
    pcurl_slist_t m_headers = nullptr;

public:

    QCurl();
    ~QCurl();

    void init();

    template<typename T>
    void set_curl_handle(CURLoption option, T data)
    {
        CURLcode res = curl_easy_setopt(m_curl_handle, option, data);
        if (res != CURLE_OK)
            throw std::runtime_error(curl_easy_strerror(res));

        return;
    }

    void set_curl_header(const std::string& header)
    {
        m_headers = curl_slist_append(m_headers, header.c_str());
        if (!m_headers)
            throw std::runtime_error("Failed to add header");

        return;
    }

    void update_curl_header(const std::string& header_key, const std::string& header_value)
    {
        pcurl_slist_t current = m_headers;
        pcurl_slist_t updated_headers = nullptr;

        while (current != nullptr) 
        {
            if (std::string(current->data).find(header_key) != std::string::npos) 
                updated_headers = curl_slist_append(updated_headers, header_value.c_str());
            else
                updated_headers = curl_slist_append(updated_headers, current->data);
            
            current = current->next;
        }

        curl_slist_free_all(m_headers);
        m_headers = updated_headers;
        set_curl_handle(CURLOPT_HTTPHEADER, m_headers);
    }

    std::string get_response_body()
    {
        return m_response_body;
    }

    //void get(const std::string url, const std::string& json);
    void post(const std::string url, const std::string& json);

private:

    std::string m_response_body;

    const size_t m_reperform_time = 3;
};

QPANDA_END

#endif
