/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _READWIRTE_LOCK_H
#define _READWIRTE_LOCK_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class SharedMutex
{
private:
    std::mutex m_mutex;
    std::condition_variable m_cond;

    bool m_is_w = false;

    size_t m_read_c = 0;

    bool read_cond() const
    {
        return false == m_is_w;
    }

    bool write_cond() const
    {
        return false == m_is_w && 0 == m_read_c;
    }

public:
    void read()
    {
       std:: unique_lock<std::mutex> lck(m_mutex);

        m_cond.wait(lck, std::bind(&SharedMutex::read_cond, this));
        m_read_c++;
    }

    void unread()
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        m_read_c--;
        m_cond.notify_all();
    }

    void write()
    {
        std::unique_lock<std::mutex> lck(m_mutex);

        m_cond.wait(lck, std::bind([](const bool *is_w, const size_t *read_c) -> bool
        {
            return false == *is_w && 0 == *read_c;
        }, &m_is_w, &m_read_c));
        m_is_w = true;
    }

    void unwrite()
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        m_is_w = false;
        m_cond.notify_all();
    }
};

class ReadLock
{
private:
    SharedMutex * m_sm;
public:
    ReadLock(SharedMutex &sm)
    {
        m_sm = &sm;
        m_sm->read();
    }
    ~ReadLock()
    {
        m_sm->unread();
    }
};

class WriteLock
{
private:
    SharedMutex * m_sm;
public:
    WriteLock(SharedMutex &sm)
    {
        m_sm = &sm;
        m_sm->write();
    }
    ~WriteLock()
    {
        m_sm->unwrite();
    }
};


#endif // !_READWIRTE_LOCK_H


