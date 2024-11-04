#ifndef TCP_MSG_H
#define TCP_MSG_H

#include "ELog.h"

namespace TCPMsg {
#define MSG_HEAD_LEN   4  
#define MSG_HEAD_FLAG   ("***<")
#define CRC_FLAG_LEN   4           /* 数据校验位长度 */

#pragma pack(push,1)
	struct TcpMsgHead
	{
		unsigned char m_head_flag[MSG_HEAD_LEN];
		uint16_t m_type;
		uint16_t m_sn_id;       /**< 消息流水ID */
		uint32_t m_body_len;    /**< 消息净荷长度 */
	};
#pragma pack(pop)

	enum class TcpMsgType : uint16_t
	{
		UNKNOW_MSG_TYPE = 0,
		STATE_MSG = 1,
		RESULT_MSG,
		TASK_ID_MSG,
		RESULT_ACK_MSG
	};

	struct TcpMsg
	{
		TcpMsgHead m_head;
		char* m_msg_body;
		//uint8_t m_crc[CRC_FLAG_LEN];
	};

	template<typename _Ty>
	inline std::shared_ptr<_Ty> _make_arry_shared(const uint32_t& len) {
		// make a shared_ptr
		//std::cout << "On new memery_len: " << len << std::endl;
		return std::shared_ptr<_Ty>(new(std::nothrow) _Ty[len], [len](_Ty* p) {
			//std::cout << "On del memery_len: " << len << std::endl;
            delete[] p; });
	}

	struct TcpMsgBuilder
	{
		std::shared_ptr<uint8_t> m_p_msg;
		const uint32_t m_msg_len;
		const uint16_t m_msg_type;

		TcpMsgBuilder(std::string str, TcpMsgType msg_type)
			:m_msg_len(sizeof(TcpMsgHead) + str.size() + CRC_FLAG_LEN)
			, m_msg_type((uint16_t)msg_type)
		{
			m_p_msg = _make_arry_shared<uint8_t>(m_msg_len);

			TcpMsgHead* p_head = (TcpMsgHead*)(m_p_msg.get());
			memcpy(p_head->m_head_flag, MSG_HEAD_FLAG, strlen(MSG_HEAD_FLAG));
			p_head->m_type = m_msg_type;
			p_head->m_sn_id = 0;
			p_head->m_body_len = str.size();
			memcpy((unsigned char*)m_p_msg.get() + sizeof(TcpMsgHead), str.c_str(), str.size());

			unsigned char* p_crc = (unsigned char*)m_p_msg.get() + sizeof(TcpMsgHead) + str.size();
			for (size_t i = 0; i < CRC_FLAG_LEN; ++i) {
				p_crc[i] = ((p_head->m_body_len >> (i * 8)) & 0xFF);
			}
		}
	};

	inline bool verify_msg_crc(const TcpMsg* msg)
	{
		uint32_t recv_crc = 0;
		TcpMsgHead* p_head = (TcpMsgHead*)msg;
		unsigned char* p_crc = (unsigned char*)msg + sizeof(TcpMsgHead) + p_head->m_body_len;
		for (size_t i = 0; i < CRC_FLAG_LEN; ++i) {
			recv_crc += ((uint8_t)(p_crc[i])) << (i * 8);
		}

		if (msg->m_head.m_body_len == recv_crc) {
			return true;
		}

        PTraceError("Error: Crc failed, recv crc_val=" << recv_crc << 
            ", target_crc_val=" << msg->m_head.m_body_len);
		return false;
	}
}

#endif // !TCP_MSG_H
