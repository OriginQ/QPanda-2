#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <vector>
#include <queue>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <condition_variable>
#include <memory>
#include <functional>
#include "Core/Utilities/Tools/QPandaException.h"
#include <stdlib.h>
#include <atomic>

QPANDA_BEGIN

#define MAX_THREADS 1024
#define DEFAULT_THREAD_CNT 8

typedef std::function<void(void)> Task;

class threadPool
{
public:
	threadPool()
		:m_stop(false), m_init_ok(false)
	{}
	~threadPool() {
		m_stop = true;
		m_condition.notify_all();
		for (auto &w : m_work_threads)
		{
			w.join();
		}
	}

	/*!
	* @brief  init thread pool
	* @param[in]  size_t Number of threads in the thread pool, 8 by default 
	* @return     void
	*/
	bool init_thread_pool(size_t thread_cnt = DEFAULT_THREAD_CNT) {
		if (thread_cnt <= 0 || thread_cnt > MAX_THREADS)
		{
			QCERR_AND_THROW_ERRSTR(init_fail, "Error: The max-thread-number is 1024.");
		}
		for (int i = 0; i < thread_cnt; ++i)
		{
			m_work_threads.emplace_back(std::bind(&threadPool::run, this));
		}
		m_init_ok = true;
		return m_init_ok;
	}

	bool append(Task task) {
		if (!m_init_ok){
			QCERR_AND_THROW(run_fail, "Error: Failed to append task, please initialize the threadPool first.");
		}

		m_queue_mutex.lock();
		m_tasks_queue.push(task);
		m_queue_mutex.unlock();

		m_condition.notify_all();
		return true;
	}

private:
	void run() {
		while (!m_stop)
		{
			Task tmp_task = nullptr;

			{
				std::unique_lock<std::mutex> lk(m_queue_mutex);
				m_condition.wait_for(lk, std::chrono::milliseconds(100), [this]() { return !m_tasks_queue.empty(); });
				if (m_tasks_queue.empty())
				{
					continue;
				}
				else
				{
					tmp_task = m_tasks_queue.front();
					m_tasks_queue.pop();
				}
			}

			if (nullptr != tmp_task)
			{
				tmp_task();
			}
		}
	}

private:
	std::vector<std::thread> m_work_threads;
	std::queue<Task> m_tasks_queue;
	std::mutex m_queue_mutex;
	std::condition_variable m_condition;
	std::atomic<bool> m_stop;
	bool m_init_ok;
};

QPANDA_END
#endif