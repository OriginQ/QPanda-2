#ifndef ASYNC_TASK_H
#define ASYNC_TASK_H

#include <future>
#include <functional>
#include "Core/Utilities/QPandaNamespace.h"

/**
 * @brief template tool for get return type of function
 */
template <typename _Func>
struct FuncRet;

template <typename _Ret, typename... _Args>
struct FuncRet<_Ret (*)(_Args...)>
{
    using type = _Ret;
};

template <typename _Ret, typename _Cls, typename... _Args>
struct FuncRet<_Ret (_Cls::*)(_Args...)>
{
    using type = _Ret;
};

/**
 * @brief special forward for std::async 
 * the arguments to the thread function are moved or copied by value.
 * If a reference argument needs to be passed to the thread function,
 * it has to be wrapped(e.g std::ref)
 * 
 * this template functions work for reference wrap
 * 
 * @warning pass reference to thread function without thread lock may cause data racing
 * it's better do not do that
 */
template <typename _T, typename is_lvalue_reference = void>
struct async_forward
{
    static _T forward(_T t)
    {
        return std::move(t);
    }
};

template <typename _T>
struct async_forward<_T, typename std::enable_if<std::is_lvalue_reference<_T>::value>::type>
{
    static auto forward(_T t) -> decltype(std::ref(t))
    {
        return std::ref(t);
    }
};

QPANDA_BEGIN

template <typename _TaskFunc, typename _TraceFunc>
class BaseAsyncTask
{
public:
    using _TaskRet = typename FuncRet<_TaskFunc>::type;
protected:
    _TaskFunc m_task_func;
    _TraceFunc m_trace_func;
    std::future<_TaskRet> m_async_thread;

public:
    BaseAsyncTask() = delete;
    BaseAsyncTask(const BaseAsyncTask &) = delete;
    BaseAsyncTask &operator=(const BaseAsyncTask &) = delete;

    BaseAsyncTask(_TaskFunc task_func, _TraceFunc trace_func)
        : m_task_func(task_func), m_trace_func(trace_func) {}

    virtual ~BaseAsyncTask() {}

    virtual void wait() const
    {
        if (m_async_thread.valid())
        {
            m_async_thread.wait();
        }
        else
        {
            return;
        }
    }

    virtual bool is_finished()
    {
        // after future::get, call future::wait will cause future state error
        if (m_async_thread.valid())
        {
            return m_async_thread.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }
        else
        {
            return true;
        }
    }

    virtual _TaskRet result()
    {
        return m_async_thread.get();
    }


};

/**
 * @brief general purpose async task
 * there are too many ways call function(e.g. member func, lambda, func obj)
 * to simplify async task creating and run, we treat all functions as C function style,
 * we also support user defined process trace of async task
 * 
 * @tparam _TaskFunc    task function ptr, any function type is ok
 * @tparam _TraceFunc   user defined async task process get func ptr
 */
template <typename _TaskFunc, typename _TraceFunc>
class AsyncTask;

/**
 * @brief specialization for static member func or genernal func
 */
template <typename _Ret_t, typename... _Args_t, typename _TraceFunc>
class AsyncTask<_Ret_t (*)(_Args_t...), _TraceFunc> : public BaseAsyncTask<_Ret_t (*)(_Args_t...), _TraceFunc>
{
private:
    using _TaskFunc = _Ret_t (*)(_Args_t...);
    using _Base = BaseAsyncTask<_Ret_t (*)(_Args_t...), _TraceFunc>;
    using _TraceRet = typename FuncRet<_TraceFunc>::type;
    using _BaseTaskRet = typename _Base::_TaskRet;
public:
    AsyncTask(_TaskFunc task_func, _TraceFunc trace_func)
        : _Base(task_func, trace_func) {}
    virtual ~AsyncTask() {}

    /**
     * @brief start async task
     * 
     * @param[inout] args args used by task_func
     */
    void run(_Args_t... args)
    {
        std::function<_BaseTaskRet(_Args_t...)> task = _Base::m_task_func;
        _Base::m_async_thread = std::async(std::launch::async, task, async_forward<_Args_t>::forward(args)...);
    }

    /**
     * @brief get the async task process
     * 
     * @note 1st args should be this ptr of class obj if trace func is a member func
     */
    template <typename... _Args>
    _TraceRet get_process(_Args &&...args)
    {
        std::function<_TraceRet(_Args...)> trace = _Base::m_trace_func;
        return trace(std::forward<_Args>(args)...);
    }
};

/**
 * @brief specialization for static member func or genernal func
 */
template <typename _Ret_t, typename _Cls_t, typename... _Args_t, typename _TraceFunc>
class AsyncTask<_Ret_t (_Cls_t::*)(_Args_t...), _TraceFunc> : public BaseAsyncTask<_Ret_t (_Cls_t::*)(_Args_t...), _TraceFunc>
{
private:
    using _TaskFunc = _Ret_t (_Cls_t::*)(_Args_t...);
    using _Base = BaseAsyncTask<_Ret_t (_Cls_t::*)(_Args_t...), _TraceFunc>;
    using _TraceRet = typename FuncRet<_TraceFunc>::type;
    using _BaseTaskRet = typename _Base::_TaskRet;
public:
    AsyncTask(_TaskFunc task_func, _TraceFunc trace_func)
        : _Base(task_func, trace_func) {}
    virtual ~AsyncTask() {}

    /**
     * @brief start async task
     * 
     * @param[inout] args args used by task_func
     */
    void run(_Cls_t* obj, _Args_t... args)
    {
        std::function<_BaseTaskRet(_Cls_t*, _Args_t...)> task = _Base::m_task_func;
        _Base::m_async_thread = std::async(std::launch::async, task, obj, async_forward<_Args_t>::forward(args)...);
    }

    void run(_Cls_t& obj, _Args_t... args)
    {
        std::function<_BaseTaskRet(_Cls_t&, _Args_t...)> task = _Base::m_task_func;
        _Base::m_async_thread = std::async(std::launch::async, task, obj, async_forward<_Args_t>::forward(args)...);
    }

    /**
     * @brief get the async task process
     * 
     * @note 1st args should be this ptr of class obj if trace func is a member func
     */
    template <typename... _Args>
    _TraceRet get_process(_Args &&...args)
    {
        std::function<_TraceRet(_Args...)> trace = _Base::m_trace_func;
        return trace(std::forward<_Args>(args)...);
    }
};

QPANDA_END

#endif