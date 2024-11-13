#ifndef TCP_CLIENT_H
#define TCP_CLIENT_H

#include "Core/Utilities/QPandaNamespace.h"

#if defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

#define INVALID_SOCKET -1
typedef int Socket_t;
#define Sleep(interval)	usleep(interval*1000)
#define closesocket(socket_id) close(socket_id)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <winsock.h>
#include <windows.h>
#include <process.h>
#pragma comment(lib, "wsock32.lib")
typedef SOCKET Socket_t;
typedef int socklen_t;
#endif

#include "TCPMsg.h"
#include "ELog.h"
#include "OSDef.h"

QPANDA_BEGIN

#define MAX_BUF_SIZE (1024*1024*512)    /**< buf 空间 1024*1024*512 : 512M */

class TCPClient
{
public:
	TCPClient() {}
	~TCPClient() {
		stop_heart_thread();

		close_socket();
	}

	bool init(const char* ip, const unsigned short& port, string task_id)
	{
		m_b_stop = false;
		m_task_id = task_id;
		m_socket_id = socket(AF_INET, SOCK_STREAM, 0);
		struct sockaddr_in server_addr;
		server_addr.sin_family = AF_INET;
		server_addr.sin_addr.s_addr = inet_addr(ip);
		server_addr.sin_port = htons(port);

		//int buffer = 1024;
		/*auto ret = setsockopt(m_socket_id, SOL_SOCKET, SO_RCVBUF, (void*)&buffer, sizeof(int));
		if (ret < 0)
		{
			return ret;
		}*/

		auto ret = connect(m_socket_id, (struct sockaddr*)&server_addr, sizeof(server_addr));
		if (ret < 0){
			PTraceError("connect error.");
			close_socket();
			return false;
		}

		if (m_socket_id >= 0)
		{
#if defined(_WIN32) || defined(_WIN64)
			int recv_timeout = 3000;
			socklen_t recv_len = sizeof(recv_timeout);
			auto ret = setsockopt(m_socket_id, SOL_SOCKET, SO_RCVTIMEO, (char*)&recv_timeout, recv_len);
			if (ret != 0)
			{
				PTraceError("Error: PilotOS client socket setsocketopt SO_RCVTIMEO error.");
				close_socket();
				return false;
			}
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
			struct timeval recv_timeout = { .tv_sec = 3,.tv_usec = 0 };
			socklen_t recv_len = sizeof(recv_timeout);
			auto ret = setsockopt(m_socket_id, SOL_SOCKET, SO_RCVTIMEO, (void*)&recv_timeout, recv_len);
			if (ret < 0)
			{
				PTraceError("Error: PilotOS client socket setsocketopt SO_RCVTIMEO error.");
				close_socket();
				return false;
			}
#endif
		}
		else
		{
			PTraceError("Error: PilotOS machine create socket error.");
			return false;
		}

		return true;
	}

	void on_got_task_result()
	{
		const auto ret = send_data(m_task_id, TCPMsg::TcpMsgType::RESULT_ACK_MSG);
		if (ret != (m_task_id).size() + sizeof(TCPMsg::TcpMsgHead) + CRC_FLAG_LEN) {
			PTraceError("tcp send task_result_ack failed: " << ret << "B, taskid=" << m_task_id);
		}
	}

	void heart()
	{
		const uint32_t time_slice_cnt = 30;
		while (!m_b_stop)
		{
			try
			{
				if (!m_task_id.empty())
				{
					//std::lock_guard<std::mutex> _guard(g_send_mutex);
					PTraceInfo("On heart for task: " << m_task_id);
					const auto ret = send_data(m_task_id, TCPMsg::TcpMsgType::TASK_ID_MSG);
					if (ret != (m_task_id).size() + sizeof(TCPMsg::TcpMsgHead) + CRC_FLAG_LEN) {
						PTraceError("tcp send taskID failed: " << ret << "B, taskid=" << m_task_id);
						break;
					}
				}
				else
				{
					PTraceError("Task if empty!");
					break;
				}
			}
			catch (...) {
				PTraceError("Unknow error.");
			}

			for (size_t _i = 0; (_i < time_slice_cnt) && (!m_b_stop); ++_i){
				this_thread::sleep_for(std::chrono::milliseconds(70));
			}
		}

        PTraceInfo("On heart-thread exit.");
	}

	/* Note: 返回值为消息净荷长度
	 */
	int full_recv(Socket_t s, char* buf, int buf_len, int flags)
	{
		int total_recv_size = 0;
		uint32_t need_recv_size = sizeof(TCPMsg::TcpMsgHead) + CRC_FLAG_LEN; /**< 最小接收长度 */
		uint32_t full_pkg_size = 0;
		int msg_body_size = 0;
		while (true)
		{
			const auto tmp_recv_size = recv(m_socket_id, buf + total_recv_size, need_recv_size, flags);
			if (tmp_recv_size <= 0) 
			{
#if defined(_WIN32) || defined(_WIN64)
				const auto error_num = WSAGetLastError();
				if (error_num == 10060){
					continue;
				}
                //PTraceWran("Warn: recv on windows error_num: " << error_num);
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
				const auto error_num = errno;
                //PTraceError("Warn: recv on linux error_num: " << error_num);
				if (error_num == 35) {
					continue;
				}
#endif
                PTraceWarn("Warn: recv on error_num: " << error_num);
				break;
			}

            PTraceInfo("On full_recv tmp_recv " << tmp_recv_size << " B");
			total_recv_size += tmp_recv_size;

			if (full_pkg_size == total_recv_size){
                PTraceInfo("Recv full_recv tmp_recv " << tmp_recv_size << " B");
				break;
			}

			if ((0 == full_pkg_size) && (sizeof(TCPMsg::TcpMsgHead) < total_recv_size))
			{
				TCPMsg::TcpMsgHead* p_head = (TCPMsg::TcpMsgHead*)buf;
				msg_body_size = p_head->m_body_len;

				/* 剩余要接收的数据=包头+净荷+CRC校验-已接收 */
				full_pkg_size = sizeof(TCPMsg::TcpMsgHead) + msg_body_size + CRC_FLAG_LEN;
			}

			need_recv_size = full_pkg_size - total_recv_size;
		}

        PTraceInfo("On full_recv total recv " << total_recv_size << " B, msg-body size=" << msg_body_size);
		return msg_body_size;
	}

	int send_data(const string& data_str, TCPMsg::TcpMsgType msg_type)
	{
		TCPMsg::TcpMsgBuilder msg_builder(data_str, msg_type);
		while (true)
		{
			const auto _send_size = send(m_socket_id, (const char*)msg_builder.m_p_msg.get(), msg_builder.m_msg_len, 0);
			if (_send_size != msg_builder.m_msg_len)
			{
#if defined(_WIN32) || defined(_WIN64)
				const auto error_num = WSAGetLastError();
                PTraceError("Error: send data error: " << error_num << ", send return: " << _send_size
					<< ", need send_size=" << msg_builder.m_msg_len);
				/*if (error_num == 10060) {
					continue;
				}*/
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
                PTraceError("Error: send data error: " << errno << ", send return: " << _send_size
					<< ", need send_size=" << m_task_id.size());
				if (errno == 9) { /* connection closed */
					break;
				}
#endif
				this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}

			break;
		}

		return msg_builder.m_msg_len;
	}

	void run_heart_thread(){
		m_thread_heart = thread(std::bind(&TCPClient::heart, this));
	}

	void stop_heart_thread()
	{
		m_b_stop = true;
		if (m_thread_heart.joinable()) {
			this_thread::sleep_for(std::chrono::milliseconds(20));
			m_thread_heart.join();
		}
	}

	void close_socket()
	{
		if (-1 != m_socket_id){
			PTraceInfo("On close socket : " << m_socket_id);
			closesocket(m_socket_id);
			m_socket_id = -1;
		}
	}

	void wait_for_close()
	{
		const auto start = chrono::system_clock::now();
		char buf[64] = "";
		while (true)
		{
			const auto tmp_recv_size = recv(m_socket_id, buf,
				64, 0);
			if (tmp_recv_size <= 0)
			{
#if defined(_WIN32) || defined(_WIN64)
				const auto error_num = WSAGetLastError();
				PTraceError("Error: recv on windows error: " << error_num);
				if (error_num == 10060) {
					continue;
				}
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
				const auto error_num = errno;
                PTraceError("Error: recv on linux error: " << error_num);
				/*if (error_num == 10060) {
					continue;
				}*/
#endif
				break;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}

		const auto end = chrono::system_clock::now();
		const auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        PTraceInfo("The wait_for_close takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds");
	}

	bool wait_recv_task_result(std::string& recv_msg, const std::string &task_id)
	{
        PTraceInfo("recv_msg:" << recv_msg);
		m_recv_buf = std::make_unique<char []> (MAX_BUF_SIZE);//TCPMsg::_make_arry_shared<char>(MAX_BUF_SIZE);

		try
		{
			int recv_msg_body_size = 0;
			while (true)
			{
				memset(m_recv_buf.get(), 0, MAX_BUF_SIZE);
				recv_msg_body_size = full_recv(m_socket_id, m_recv_buf.get(), MAX_BUF_SIZE - 1, 0);
                if (recv_msg_body_size)
                {
                    JsonMsg::JsonParser parser;
                    if (parser.load_json(m_recv_buf.get()) && parser.has_member_string("taskId") && (parser.get_string("taskId") != task_id))
                    {
                        PTraceInfo("Wrong taskId: " << parser.get_string("taskId"));
                        return false;
                    }
                }
                PTraceInfo("tmp recv " << recv_msg_body_size << " B");

				if (recv_msg_body_size > 0)
				{
					TCPMsg::TcpMsgHead* p_msg_head = (TCPMsg::TcpMsgHead*)m_recv_buf.get();
					char* p_msg_body = (char*)(m_recv_buf.get()) + sizeof(TCPMsg::TcpMsgHead);
					
                    /* check msg type */ 
					switch ((TCPMsg::TcpMsgType)p_msg_head->m_type)
					{
					case TCPMsg::TcpMsgType::STATE_MSG:
					{
						m_task_status = string(p_msg_body, recv_msg_body_size);
                        PTraceInfo("tcp recv status: " << m_task_status << ", for task:" << m_task_id);
						const uint32_t status = std::stoul(m_task_status);
                        if ((status == ((uint32_t)(PilotQVM::TaskStatus::FAILED)))
                            || (status == ((uint32_t)(PilotQVM::TaskStatus::CANCELLED))))
                        {
                            PTraceError("Error: the status for task " << m_task_id << " is " << status);
                            //return false; /* 不退出循环，以接收具体错误信息 */
                        }
						continue;
					}
						break;

					case TCPMsg::TcpMsgType::RESULT_MSG:
					{
						stop_heart_thread();
						recv_msg.append(string(p_msg_body, recv_msg_body_size));
                        PTraceInfo(" tmp recv result size:" << recv_msg_body_size);
					}
						break;

					default:
                        PTraceError("Error: Undef msg type:" << p_msg_head->m_type);
						return false;
					}

					if (verify_msg_crc((TCPMsg::TcpMsg*)m_recv_buf.get()))
					{
                        PTraceInfo("^_^^_^^_^^_^^_^ recv good result-msg ^_^^_^^_^^_^^_^ for task: " << m_task_id);
						on_got_task_result();
						
						break;
					}
				}
				else
				{
#if defined(_WIN32) || defined(_WIN64)
					const auto error_num = WSAGetLastError();
                    PTraceError("Error: On recv failed: " << error_num);
					if (error_num != 10060)
					{
                        PTraceError("Error: Stop recv result !!!");
						//break;
					}
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
					const auto error_num = errno;
                    PTraceError("Error: On recv failed: " << error_num);
					if (error_num != 11)
					{
                        PTraceError("Error: Stop recv result !!!");
						//break;
					}
#endif
					return false;
				}
			}
		}
		catch (...)
        {
            PTraceError("Error: unknow error on recv task-result.");
            return false;
		}

		return true;
	}

private:
	Socket_t m_socket_id;
	string m_task_id;
	thread m_thread_heart;
	std::atomic<bool> m_b_stop{false};
	std::string m_task_status;
	//std::shared_ptr<char> m_recv_buf;
	std::unique_ptr<char []> m_recv_buf;
};

QPANDA_END

#endif // !TCP_CLIENT_H
