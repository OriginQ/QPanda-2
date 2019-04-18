/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

marco.h

Author: LiYe
Created in 2018-11-08


*/
#ifndef MARCO_H
#define MARCO_H

#define    SINGLETON_DECLARE(Type)            \
public:                                       \
    static Type*   getInstance();             \
    static void    release();                 \
protected:                                    \
    static Type*   m_instance;

#define    SINGLETON_IMPLEMENT_EAGER(Type)    \
    Type*  Type::m_instance = new Type;       \
    Type*  Type::getInstance()                \
    {                                         \
        if (!m_instance)                      \
        {                                     \
            m_instance = new Type;            \
        }                                     \
        return m_instance;                    \
    }                                         \
    void   Type::release()                    \
    {                                         \
        if (m_instance)                       \
        {                                     \
           delete m_instance;                 \
        }                                     \
        m_instance = nullptr;                 \
    }

#define    SINGLETON_IMPLEMENT_LAZY(Type)     \
    Type*  Type::m_instance = nullptr;        \
    Type*  Type::getInstance()                \
    {                                         \
        if (!m_instance)                      \
        {                                     \
            m_instance = new Type;            \
        }                                     \
        return m_instance;                    \
    }                                         \
    void   Type::release()                    \
    {                                         \
        if (m_instance)                       \
        {                                     \
           delete m_instance;                 \
        }                                     \
        m_instance = nullptr;                 \
    }

#endif // MARCO_H
