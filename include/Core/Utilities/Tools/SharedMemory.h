#ifndef QPANDA_TOOLS_MEMORYSHARE
#define QPANDA_TOOLS_MEMORYSHARE

#include <mutex>
#include <string>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

class SharedMemory
{
private:
    int key = { 0 };
    int handle = { 0 };
    void* shared_memory = nullptr;
#if defined(_WIN32) || defined(_WIN64)
    void* shared_handle = nullptr;
#endif

public:
    ~SharedMemory()
    {
#if defined(_WIN32) || defined(_WIN64)
        if (shared_memory != nullptr)
        {
            UnmapViewOfFile(shared_memory);
        }
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
        if (shared_memory != nullptr)
        {
            shmdt(shared_memory);
        }
#endif
    }

    void memory_delete(void)
    {
#if defined(_WIN32) || defined(_WIN64)
        if (shared_memory != nullptr)
        {
            CloseHandle(shared_handle);
        }
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
        if (shared_memory != nullptr)
        {
            shmctl(handle, IPC_RMID, NULL);
        }
#endif
    }

    SharedMemory(size_t size, char* name)
    {
#if defined(_WIN32) || defined(_WIN64)
        if (size > 0 && name != nullptr)
        {
            if ((shared_handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, name)) == nullptr)
            {
                if ((shared_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, name)) != nullptr)
                {
                    if ((shared_memory = MapViewOfFile(shared_handle, FILE_MAP_ALL_ACCESS, 0, 0, 0)) == nullptr)
                    {
                        throw std::runtime_error("shared memory error:MapViewOfFile");
                    }
                    memset(shared_memory, 0, size);
                }
                else
                {
                    throw std::runtime_error("shared memory error:CreateFileMappingW");
                }
            }
            else
            {
                if ((shared_memory = MapViewOfFile(shared_handle, FILE_MAP_ALL_ACCESS, 0, 0, 0)) == nullptr)
                {
                    throw std::runtime_error("shared memory error:MapViewOfFile");
                }
            }
        }
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
        if (size > 0 && name != nullptr)
        {
            while (*name != '\0')
            {
                key = (key * 33 + *name) & 0xFFFF;
                name += 1;
            }
            if ((handle = shmget(key, size, IPC_EXCL)) == -1)
            {
                if ((handle = shmget(key, size, IPC_CREAT | 0666)) == -1)
                {
                    throw std::runtime_error("shared memory error:shmget");
                }
                if (size_t(shared_memory = shmat(handle, NULL, SHM_R | SHM_W | 0666)) == -1)
                {
                    throw std::runtime_error("shared memory error:shmat error");
                }
                memset(shared_memory, 0, size);
            }
            else
            {
                if (size_t(shared_memory = shmat(handle, NULL, SHM_R | SHM_W | 0666)) == -1)
                {
                    throw std::runtime_error("shared memory error:shmat");
                }
            }
            }
        else
        {
            throw std::runtime_error("shared memory error:invalid parms");
        }
#endif
    }

    void*& memory(void)
    {
        return shared_memory;
    }
};

#endif