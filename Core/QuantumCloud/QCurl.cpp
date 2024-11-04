#include <thread>
#include <sstream>
#include <iostream>
#include "Core/QuantumCloud/QCurl.h"
#include "Core/QuantumCloud/QCloudLog.h"

USING_QPANDA
using namespace std;

#if defined(USE_CURL)

static size_t write_call_back(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));

    *((std::stringstream*)stream) << data << std::endl;

    return size * nmemb;
}

void QCurl::init(std::string user_token)
{
    std::string auth = "Authorization: oqcs_auth=" + user_token;

    set_curl_header(auth);
    set_curl_header("Content-Type: application/json");
    set_curl_header("Connection: keep-alive");
    //set_curl_header("Transfer-Encoding: chunked");
    set_curl_header("origin-language: en");

    set_curl_handle(CURLOPT_HTTPHEADER, m_headers);
    set_curl_handle(CURLOPT_TIMEOUT, 60);
    set_curl_handle(CURLOPT_CONNECTTIMEOUT, 30);
    set_curl_handle(CURLOPT_HEADER, 0);
    set_curl_handle(CURLOPT_SSL_VERIFYHOST, 0);
    set_curl_handle(CURLOPT_SSL_VERIFYPEER, 0);
    //set_curl_handle(CURLOPT_VERBOSE, 1L);

    set_curl_handle(CURLOPT_READFUNCTION, nullptr);
    set_curl_handle(CURLOPT_NOSIGNAL, 1);
    set_curl_handle(CURLOPT_WRITEFUNCTION, write_call_back);

    return;
}

QCurl::QCurl()
{
    curl_global_init(CURL_GLOBAL_ALL); 

    m_curl_handle = curl_easy_init();

    if (!m_curl_handle)
        throw std::runtime_error("curl_easy_init failed");
}

QCurl::~QCurl()
{
    if (m_curl_handle)
        curl_easy_cleanup(m_curl_handle);

    if (m_headers)
        curl_slist_free_all(m_headers);

    curl_global_cleanup();
}

void QCurl::post(const std::string url, const std::string& json)
{
    std::stringstream stream;
    set_curl_handle(CURLOPT_POST, 1);
    set_curl_handle(CURLOPT_URL, url.c_str());
    set_curl_handle(CURLOPT_WRITEDATA, &stream);
    set_curl_handle(CURLOPT_POSTFIELDS, json.c_str());
    set_curl_handle(CURLOPT_POSTFIELDSIZE, json.size());

    CURLcode curl_res_code;
    for (size_t i = 0; i < m_reperform_time; i++)
    {
        QCLOUD_LOG_INFO("curl perform url : " + url);
        QCLOUD_LOG_INFO("curl perform json : " + json);

        curl_res_code = curl_easy_perform(m_curl_handle);
        if (CURLE_OK == curl_res_code) 
        {
            m_response_body = stream.str();

            QCLOUD_LOG_INFO(m_response_body);

            int pos = 0;
            while ((pos = m_response_body.find("\n")) != -1)
                m_response_body.erase(pos, 1);

            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (CURLE_OK != curl_res_code)
    {
        std::string err_code_msg = curl_easy_strerror(curl_res_code);
        throw std::runtime_error("curl_easy_perform() failed: " + err_code_msg);
    }
}

void QCurl::get(const std::string url)
{
    std::stringstream stream;
    set_curl_handle(CURLOPT_URL, url.c_str());
    set_curl_handle(CURLOPT_HTTPGET, 1L);
    set_curl_handle(CURLOPT_WRITEDATA, &stream);

    CURLcode curl_res_code;
    for (size_t i = 0; i < m_reperform_time; i++)
    {
        QCLOUD_LOG_INFO("curl perform url : " + url);

        curl_res_code = curl_easy_perform(m_curl_handle);
        if (CURLE_OK == curl_res_code)
        {
            m_response_body = stream.str();
            QCLOUD_LOG_INFO(m_response_body);

            int pos = 0;
            while ((pos = m_response_body.find("\n")) != -1)
                m_response_body.erase(pos, 1);

            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (CURLE_OK != curl_res_code)
    {
        std::string err_code_msg = curl_easy_strerror(curl_res_code);
        throw std::runtime_error("curl_easy_perform() failed: " + err_code_msg);
    }
}

#endif
