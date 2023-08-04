#ifndef E_LOG_H
#define E_LOG_H

#include "./easylogging++.h"
#include "Core/Utilities/QPandaNamespace.h"

namespace PilotQVM {
	class ELog
	{
	public:
		static ELog& get_instance() {
			static ELog _instance;
			return _instance;
		}
		~ELog() {};

		void set_output_log(const bool b) {
			m_b_output_log = b;
		}

		bool get_output_log() {
			return m_b_output_log.load();
		}

	protected:
		ELog() {};

	private:
		std::atomic_bool m_b_output_log{ false };
	};

#define PTraceInfo(_msg){if (PilotQVM::ELog::get_instance().get_output_log()){LOG(INFO) << __FILENAME__ << " " << __LINE__ << ": " << _msg;}}
#define PTraceError(_msg){if (PilotQVM::ELog::get_instance().get_output_log()){LOG(ERROR) << __FILENAME__ << " " << __LINE__ << ": " <<_msg;}}
#define PTraceDebug(_msg){if (PilotQVM::ELog::get_instance().get_output_log()){LOG(DEBUG) << __FILENAME__ << " " << __LINE__ << ": " <<_msg;}}
#define PTraceWarn(_msg){if (PilotQVM::ELog::get_instance().get_output_log()){LOG(WARNING) << __FILENAME__ << " " << __LINE__ << ": " <<_msg;}}
}

#endif // !E_LOG_H
