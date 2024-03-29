#ifndef _ORIGIN_COLLECTION_H
#define _ORIGIN_COLLECTION_H 
#include <string>
#include <ctime>
#include <vector>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <utility>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/reader.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/istreamwrapper.h"
#include "ThirdParty/rapidjson/ostreamwrapper.h"
#include <type_traits>

#include <codecvt>
#include <string>
#include <sys/stat.h>

QPANDA_BEGIN

#if defined(WIN32) || defined(_WIN32)
#define localtime_r(_Time, _Tm) localtime_s(_Tm, _Time)
#endif
using Value = rapidjson::Value;

/**
 * @brief Origin Collection
 * A relatively free data collection class for saving data
 * @tparam number The num of key 
 */
class OriginCollection
{
private:
    std::vector<std::string> m_key_vector; ///< key vector
    rapidjson::Document m_doc;             ///< json doc
    std::string m_file_path;
    std::string m_db_dir = "QPanda_DB";
    std::string m_db_bp = "bplus_tree";
    size_t m_file_count;
    template<typename T>
    void valuePushBack(Value &value_json,T & value )
    {
        if (std::is_same<typename std::decay<T>::type, int>::value)
        {
            value_json.PushBack((int)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, unsigned>::value)
        {
            value_json.PushBack((unsigned)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, int64_t>::value)
        {
            value_json.PushBack((int64_t)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, uint64_t>::value)
        {
            value_json.PushBack((uint64_t)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, double>::value)
        {
            value_json.PushBack((double)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, float>::value)
        {
            value_json.PushBack((float)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, bool>::value)
        {
            value_json.PushBack((bool)value, m_doc.GetAllocator());
        }
        else if (std::is_same<typename std::decay<T>::type, size_t>::value)
        {
            value_json.PushBack((rapidjson::SizeType)value, m_doc.GetAllocator());
        }
        else 
        {
            QCERR("param error");
            throw std::invalid_argument("param error");    
	    }
    }
    /**
     * @brief Check if the input name are valid 
     * @param[in] name target name 
     */
    inline std::vector<std::string>::iterator checkVaild(const std::string & name)
    {
        bool result = false;
        auto aiter = m_key_vector.begin();
        for(;aiter!= m_key_vector.end();++aiter)
        {
            if(name == *aiter)
            {
                result = true;
                break;
            }
        }

        if(false == result)
        {
            QCERR("The name entered is not in the m_key_vector");
            throw std::invalid_argument("The name entered is not in the m_key_vector");
        }
        if(m_doc.HasMember(name.c_str()))
        {
            m_doc.RemoveMember(name.c_str());
        }
        return aiter;
    }

    /**
    * @brief add value
    * Set the value corresponding to the key
    * @tparam T Value type
    * @tparam ARG Variable length parameter tparam
    * @param key_name Key name
    * @param num Key position in json
    * @param value Key value
    * @param arg Variable length parameter
    */
    template<typename T, typename... ARG>
    void addValue(const std::string & key_name, const T value, ARG... arg)
    {
        int i = -1;
        for (size_t num = 0; num < m_key_vector.size(); num++)
        {
            if (m_key_vector[num] == key_name)
            {
                i = (int)num;
                break;
            }
        }
        if (i == -1)
        {
            return;
        }
        addValue(key_name, value);

        if (i < m_key_vector.size() - 1)
        {
            addValue(m_key_vector[i + 1], arg...);
        }
    }

    void addValue(const std::vector<std::string> & key_vector)
    {
        for (auto aiter : m_key_vector)
        {
            auto iter = std::find(key_vector.begin(), key_vector.end(), aiter);
            if (iter == key_vector.end())
            {
                addValue(aiter);
            }
        }
    }

    template<typename T>
    void addValue(const std::vector<std::string> & key_vector, const int key_num, const T value)
    {
        if (key_num >= key_vector.size())
        {
            return;
        }
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_vector[key_num]);
        if (iter != m_key_vector.end())
        {
            addValue(*iter, value);
            addValue(key_vector);
        }
        else
        {
            QCERR("key_vector element is not an element in m_key_vector");
            throw std::runtime_error("key_vector element is not an element in m_key_vector");
        }
    }

    template<typename T, typename... ARG>
    void addValue(const std::vector<std::string> & key_vector, const int key_num, const T value, ARG... arg)
    {
        if (key_num >= key_vector.size())
        {
            return;
        }
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_vector[key_num]);
        if (iter != m_key_vector.end())
        {
            addValue(*iter, value);
            addValue(key_vector, key_num + 1, arg...);
        }
        else
        {
            QCERR("key_vector element is not an element in m_key_vector");
            throw std::runtime_error("key_vector element is not an element in m_key_vector");
        }
    }

public:

    /**
    * @brief add value
    * Assign a value to the specified key
    * @tparam T value type
    * @param value_name Key name
    * @param num Key position in json
    * @param value key value
    */
    template<typename T>
    void addValue(const std::string & key_name,const T & value)
    {
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_name);
        if (iter == m_key_vector.end())
        {
            return;
        }
        auto &allocator = m_doc.GetAllocator();
        if (m_doc.HasMember(key_name.c_str()))
        {
            auto & value_json = m_doc[key_name.c_str()];
            valuePushBack(value_json,value);
//            value_json.PushBack(value, allocator);
        }
        else
        {
            Value value_json(rapidjson::kArrayType);
            valuePushBack(value_json,value);
            //value_json.PushBack(value, allocator);
            m_doc.AddMember(Value().SetString(key_name.c_str(), allocator).Move(),
                value_json, allocator);
        }
    }

    /**
    * @brief add value
    * Assign a value to the specified key
    * @tparam T value type
    * @param value_name Key name
    * @param num Key position in json
    * @param value key value
    */
    template<typename T>
    void addValue(const std::string & key_name,const std::vector<T> & value)
    {
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_name);
        if (iter == m_key_vector.end())
        {
            return;
        }
        auto &allocator = m_doc.GetAllocator();
        if (m_doc.HasMember(key_name.c_str()))
        {
            Value temp(rapidjson::kArrayType);
            for (auto &aiter : value)
            {
                valuePushBack(temp,aiter);
//                temp.PushBack(aiter, allocator);
            }
            auto & value_json = m_doc[key_name.c_str()];
            value_json.PushBack(temp, allocator);
        }
        else
        {
            Value value_json(rapidjson::kArrayType);
            Value temp(rapidjson::kArrayType);
            for (auto &aiter : value)
            {
                valuePushBack(temp,aiter);
                //temp.PushBack(aiter, allocator);
            }
            value_json.PushBack(temp, allocator);
            m_doc.AddMember(Value().SetString(key_name.c_str(), allocator).Move(),
                value_json, allocator);
        }
    }

    /**
    * @brief add value
    * Assign a value to the specified key
    * @tparam T value type
    * @param value_name Key name
    * @param num Key position in json
    * @param value key value
    */
    void addValue(const std::string & key_name,const std::vector<std::string> & value)
    {
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_name);
        if (iter == m_key_vector.end())
        {
            return;
        }
        auto &allocator = m_doc.GetAllocator();
        if (m_doc.HasMember(key_name.c_str()))
        {
            Value temp(rapidjson::kArrayType);
            for (auto &aiter : value)
            {
                temp.PushBack(Value().SetString(aiter.c_str(), allocator).Move(), allocator);
            }
            auto & value_json = m_doc[key_name.c_str()];
            value_json.PushBack(temp, allocator);
        }
        else
        {
            Value value_json(rapidjson::kArrayType);
            Value temp(rapidjson::kArrayType);
            for (auto &aiter : value)
            {
                temp.PushBack(Value().SetString(aiter.c_str(), allocator).Move(), allocator);
            }
            value_json.PushBack(temp, allocator);
            m_doc.AddMember(Value().SetString(key_name.c_str(), allocator).Move(),
                value_json, allocator);
        }
    }

    /**
    * @brief add value
    * Set the value corresponding to the key
    * @param value_name Key name
    * @param num Key position in json
    * @param value Key value
    */
    void addValue(const std::string & key_name,const std::string & value)
    {
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_name);
        if (iter == m_key_vector.end())
        {
            return;
        }

        auto &allocator = m_doc.GetAllocator();
        if (m_doc.HasMember(key_name.c_str()))
        {
            auto & value_json = m_doc[key_name.c_str()];
            value_json.PushBack(Value().SetString(value.c_str(), allocator).Move(),
                allocator);
        }
        else
        {
            Value value_json(rapidjson::kArrayType);
            value_json.PushBack(Value().SetString(value.c_str(), allocator).Move(),
                allocator);
            m_doc.AddMember(Value().SetString(key_name.c_str(), allocator).Move(),
                value_json, allocator);
        }
    }

    /**
    * @brief add value
    * Set the value corresponding to the key
    * @param value_name Key name
    * @param num Key position in json
    * @param value Key value
    */
    void addValue(const std::string & key_name, const char * value)
    {
        std::string value_str(value);
        addValue(key_name, value_str);
    }

    void addValue(const std::string & key_name)
    {
        auto iter = std::find(m_key_vector.begin(), m_key_vector.end(), key_name);
        if (iter == m_key_vector.end())
        {
            return;
        }

        auto &allocator = m_doc.GetAllocator();
        if (m_doc.HasMember(key_name.c_str()))
        {
            auto & value_json = m_doc[key_name.c_str()];
            value_json.PushBack(Value(rapidjson::kNullType),
                allocator);
        }
        else
        {
            Value value_json(rapidjson::kArrayType);
            value_json.PushBack(Value(rapidjson::kNullType),
                allocator);
            m_doc.AddMember(Value().SetString(key_name.c_str(), allocator).Move(),
                value_json, allocator);
        }
    }

    /**
     * @brief Construct a new Origin Collection object
     * 
     */
    inline OriginCollection()
    {
        m_doc.Parse("{}");
        m_file_count = 0;
    }

    /**
     * @brief Construct a new Origin Collection 
     * Construct a new Origin Collection by file_path
     * @param file_path File path
     */
    inline OriginCollection(const std::string & file_path,bool is_suffix = true)
    {
        m_file_count = 0;
        std::string command;
        /*
        command = "mkdir "+m_db_dir;
        auto temp = system(command.c_str());
        */
        m_file_path.append(file_path);
        if (is_suffix)
        {
            time_t now = time(nullptr);
            struct tm ltm;
            localtime_r(&now, &ltm);
            auto year = 1900 + ltm.tm_year;
            auto month = 1 + ltm.tm_mon;
            auto day = ltm.tm_mday;
            auto hour = ltm.tm_hour;
            auto min = ltm.tm_min;
            auto sec = ltm.tm_sec;

            char tmp_str[50];
            snprintf(tmp_str, sizeof(tmp_str), "%04d%02d%02d_%02d%02d%02d", year, month, day,
                hour, min, sec);
            m_file_path.append("_").append(tmp_str);
        }
        m_doc.Parse("{}");
    }
    /**
     * @brief operator=
     * Set the key of the object
     * @param args arg list Key list
     * @return OriginCollection& 
     */
    inline OriginCollection& operator=(const std::initializer_list<std::string> & args)
    {
        for(auto & aiter : args)
        {
            m_key_vector.push_back(aiter);
        }
        return *this;
    }

    /**
    * @brief operator=
    * Set the key of the object
    * @param args arg list Key list
    * @return OriginCollection&
    */
    inline OriginCollection& operator=(const std::vector<std::string> & args)
    {
        m_key_vector.resize(0);
        m_key_vector.insert(m_key_vector.end(), args.begin(), args.end());
        return *this;
    }
    /**
     * @brief Construct a new Origin Collection object by other Origin Collection
     * @param old target OriginCollection
     */
    inline OriginCollection(const OriginCollection & old)
    {
        for(auto & aiter : old.m_key_vector)
        {
            m_key_vector.push_back(aiter);
        }
        
        m_doc.CopyFrom(old.m_doc, m_doc.GetAllocator());
        m_file_count = old.m_file_count;
        m_file_path = old.m_file_path;
    }

    
    /**
     * @brief operator= by other OriginCollection
     * 
     * @param old target OriginCollection
     * @return OriginCollection& 
     */
    inline OriginCollection & operator=(const OriginCollection & old)
    {
        m_file_count = old.m_file_count;
        m_file_path = old.m_file_path;
        m_key_vector.resize(0);

        for(auto & aiter : old.m_key_vector)
        {
            m_key_vector.push_back(aiter);
        }
        m_doc.CopyFrom(old.m_doc, m_doc.GetAllocator());
        return * this;
    }
    

    /**
     * @brief Set the Value 
     * set value list by key
     * @tparam T args type
     * @param name key name
     * @param args value array
     */
    template<typename T>
    inline void setValueByKey(const std::string & name ,const std::initializer_list<T>  args)
    {
        auto iter = checkVaild(name);
        Value temp(rapidjson::kArrayType);
        auto & allocator = m_doc.GetAllocator();
        for(auto & aiter : args)
        {
            valuePushBack(temp,aiter);
            //temp.PushBack(aiter,allocator);
        }
        rapidjson::GenericStringRef<char> stringRef((*iter).c_str());
        m_doc.AddMember(stringRef,temp,allocator);
        write();
    }

    /**
     * @brief Set the Value 
     * set value list by key
     * @param name key name
     * @param args value array
     */
    inline void setValueByKey(const std::string & name, const std::initializer_list<std::string>  args)
    {
        auto iter = checkVaild(name);
        Value temp(rapidjson::kArrayType);
        auto & allocator = m_doc.GetAllocator();
        for (std::string  aiter : args)
        {
            temp.PushBack(Value().SetString(aiter.c_str(),allocator).Move(), allocator);
        }
        m_doc.AddMember(Value().SetString((*iter).c_str(), allocator).Move(), temp, allocator);
        write();
    }

    /**
     * @brief Set the Value 
     * set value list by key
     * @param name key name
     * @param args value array
     */
    inline void setValueByKey(const std::string &name, const std::initializer_list<const char *>  args)
    {
        auto iter = checkVaild(name);
        Value temp(rapidjson::kArrayType);
        auto & allocator = m_doc.GetAllocator();
        for (auto aiter : args)
        {
            temp.PushBack(Value().SetString(aiter, allocator).Move(), allocator);
        }
        m_doc.AddMember(Value().SetString((*iter).c_str(), allocator).Move(), temp, allocator);
        write();
    }

    /**
     * @brief Set the Value 
     * set vector<T> value by key
     * @tparam T args type
     * @param name key name
     * @param args value array
     */
    template<typename T>
    inline void setValueByKey(const std::string & name, const std::vector<T> & value)
    {
        checkVaild(name);
        Value temp(rapidjson::kArrayType);
        auto & allocator = m_doc.GetAllocator();
        for(auto & aiter : value)
        {
            valuePushBack(temp,aiter);
            //temp.PushBack(aiter,allocator);
        }
        rapidjson::GenericStringRef<char> stringRef(name.c_str());
        m_doc.AddMember(stringRef,temp,allocator);
        write();
    }

    /**
     * @brief insert value
     * Set the value of other properties by the value of the primary key
     * @tparam ARG  Variable length parameter tparam
     * @param key Key value
     * @param arg the value of other properties
     */
    template<class... ARG>
    inline void insertValue(const std::string & key ,ARG... arg)
    {
        if (sizeof...(arg)+1 != m_key_vector.size())
        {
            QCERR("param size is not equal to m_number");
            throw std::invalid_argument("param size count is not equal to m_number");
        }
        addValue(m_key_vector[0], key);

        addValue(m_key_vector[1],arg...);
        write();
    }

    /**
     * @brief insert value
     * Set the value of other properties by the value of the primary key
     * @tparam ARG  Variable length parameter tparam
     * @param key Key value
     * @param arg the value of other properties
     */
    template<class... ARG>
    inline void insertValue(const int key, ARG... arg)
    {
        if (sizeof...(arg) + 1 != m_key_vector.size())
        {
            QCERR("param size is not equal to m_number");
            throw std::invalid_argument("param size count is not equal to m_number");
        }
        addValue(m_key_vector[0], key);

        addValue(m_key_vector[1],arg...);
        write();
    }

    template<class... ARG>
    inline void insertValue(const std::vector<std::string> & name_vector,const std::string & key, ARG... args)
    {
        if (sizeof...(args) + 1 != name_vector.size())
        {
            QCERR("param size is not equal to name_vector size");
            throw std::invalid_argument("param size count is not name_vector size");
        }

        if (name_vector[0] != m_key_vector[0])
        {
            QCERR("name_vector[0] is not key name ");
            throw std::invalid_argument("name_vector[0] is not key name");
        }
        addValue(m_key_vector[0], key);

        addValue(name_vector, 1, args...);
        write();
    }

    template<class... ARG>
    inline void insertValue(const std::vector<std::string> & name_vector,const int key, ARG... args)
    {
        if (sizeof...(args) + 1 != name_vector.size())
        {
            QCERR("param size is not equal to name_vector size");
            throw std::invalid_argument("param size count is not name_vector size");
        }

        if (name_vector[0] != m_key_vector[0])
        {
            QCERR("name_vector[0] is not key name ");
            throw std::invalid_argument("name_vector[0] is not key name");
        }

        for (auto aiter : name_vector)
        {
            auto iter = find(m_key_vector.begin(), m_key_vector.end(), aiter);
            if (iter == m_key_vector.end())
            {
                QCERR("param error ");
                throw std::invalid_argument("param error");
            }
        }
        auto & allocator = m_doc.GetAllocator();
        addValue(m_key_vector[0], key);
        addValue(name_vector, 1, args...);
        write();
    }


    /**
     * @brief Get value by Key
     * 
     * @param name Key name
     * @return std::vector<std::string> value vector
     */
    
    inline std::vector<std::string> getValue(const std::string  name)
    {
        std::vector<std::string> value_vector;
        if(!m_doc.HasMember(name.c_str()))
        {
            QCERR("Object does not contain this name");
            return std::vector<std::string>();
            
            //throw std::invalid_argument("Object does not contain this name");
        }
        
        Value & name_value = m_doc[name.c_str()];
        
        if((name_value.IsArray()) && (!name_value.Empty()))
        {
            if(name_value[0].IsString())
            {
                for(rapidjson::SizeType i = 0 ;i<name_value.Size();i++)
                {
                    value_vector.push_back(name_value[i].GetString());
                }
            }
            else
            {
                for(rapidjson::SizeType i = 0 ;i<name_value.Size();i++)
                {
                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
                    name_value[i].Accept(write);
                    value_vector.push_back(buffer.GetString());
                }
            }

        }
        else if(!name_value.Empty())
        {
            if(name_value.IsString())
            {
                value_vector.push_back(name_value.GetString());
            }
            else
            {
                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
                    name_value.Accept(write);
                    value_vector.push_back(buffer.GetString());
            }
        }
        else
        {
            
        }
        return value_vector;
    }

    /**
     * @brief Get value by primary Key value
     * 
     * @param key_value primary Key value
     * @return std::string 
     */
    std::string getValueByKey(const std::string & key_value)
    {
        if (m_key_vector.size() <= 0)
        {
            QCERR("m_key_vector error");
            throw std::invalid_argument("m_key_vector error");
        }
        std::string temp;
        auto &value = m_doc[m_key_vector[0].c_str()];
        long long  num = 0;
        if (value.Empty())
        {
            QCERR("there is no value");
            throw std::invalid_argument("there is no value");
        }
        
        bool find_flag = false;
        for (rapidjson::SizeType i = 0; i < value.Size();i++)
        {
            if (value[i].IsString())
            {
                if (strcmp(value[i].GetString(), key_value.c_str()) == 0)
                {
                    num = (long long)i;
                    find_flag = true;
                    break;
                }
            }
        }
        
        if (!find_flag)
        {
            return temp;
        }

        Value temp_value(rapidjson::kObjectType);
        auto & allocator = m_doc.GetAllocator();
        for (size_t i = 0; i < m_key_vector.size(); i++)
        {
            rapidjson::GenericStringRef<char> stringRef(m_key_vector[i].c_str());
            temp_value.AddMember(stringRef,
                            m_doc[m_key_vector[i].c_str()][(rapidjson::SizeType)num], 
                            allocator);
        }

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
        temp_value.Accept(write);
        temp.append(buffer.GetString());
        return temp;
    }

    /**
     * @brief Get value by primary Key value
     * 
     * @param key_value primary Key value
     * @return std::string 
     */
    std::string getValueByKey(int key_value)
    {
        if (m_key_vector.size() <= 0)
        {
            QCERR("m_key_vector error");
            throw std::invalid_argument("m_key_vector error");
        }
        std::string temp;
        auto &value = m_doc[m_key_vector[0].c_str()];
        long long  num = 0;
        if (value.Empty())
        {
            QCERR("there is no value");
            throw std::invalid_argument("there is no value");
        }

        bool find_flag = false;
        for (rapidjson::SizeType i = 0; i < value.Size(); i++)
        {
            if (value[i].IsInt64())
            {
                if (value[i].GetInt64() == key_value)
                {
                    num = (long long)i;
                    find_flag = true;
                    break;
                }
            }
        }

        if (!find_flag)
        {
            return temp;
        }

        Value temp_value(rapidjson::kObjectType);
        auto & allocator = m_doc.GetAllocator();
        for (size_t i = 0; i < m_key_vector.size(); i++)
        {
            rapidjson::GenericStringRef<char> stringRef(m_key_vector[i].c_str());
            temp_value.AddMember(stringRef,
                m_doc[m_key_vector[i].c_str()][(rapidjson::SizeType)num],
                allocator);
        }

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
        temp_value.Accept(write);
        temp.append(buffer.GetString());
        return temp;
    }

    std::vector<std::string> getValueByKey(int key_value, const std::string &name)
    {
        std::vector<std::string> value_vector;
        if (m_key_vector.size() <= 0)
        {
            QCERR("m_key_vector error");
            throw std::invalid_argument("m_key_vector error");
        }

        auto &value = m_doc[m_key_vector[0].c_str()];
        long long  num = 0;
        if (value.Empty())
        {
            QCERR("there is no value");
            // throw std::invalid_argument("there is no value");
            return value_vector;
        }

        bool find_flag = false;
        for (rapidjson::SizeType i = 0; i < value.Size(); i++)
        {
            if (value[i].IsInt64())
            {
                if (value[i].GetInt64() == key_value)
                {
                    num = (long long)i;
                    find_flag = true;
                    break;
                }
            }
        }

        if (!find_flag)
        {
            return value_vector;
        }

        Value temp_value(rapidjson::kObjectType);
        auto & allocator = m_doc.GetAllocator();
        for (size_t i = 0; i < m_key_vector.size(); i++)
        {
            if (m_key_vector[i] == name)
            {
                auto &name_value = m_doc[m_key_vector[i].c_str()][(rapidjson::SizeType)num];
                if((name_value.IsArray()) && (!name_value.Empty()))
                {
                    if(name_value[0].IsString())
                    {
                        for(rapidjson::SizeType i = 0 ;i<name_value.Size();i++)
                        {
                            value_vector.push_back(name_value[i].GetString());
                        }
                    }
                    else
                    {
                        for(rapidjson::SizeType i = 0 ;i<name_value.Size();i++)
                        {
                            rapidjson::StringBuffer buffer;
                            rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
                            name_value[i].Accept(write);
                            value_vector.push_back(buffer.GetString());
                        }
                    }

                }
                else if(!name_value.Empty())
                {
                    if(name_value.IsString())
                    {
                        value_vector.push_back(name_value.GetString());
                    }
                    else
                    {
                        rapidjson::StringBuffer buffer;
                        rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
                        name_value.Accept(write);
                        value_vector.push_back(buffer.GetString());
                    }
                }

                break;
            }
        }

        return value_vector;
    }

    /**
     * @brief open
     * Read the json file of the specified path
     * @param file_name file path
     * @return true open success
     * @return false open fail
     */
    inline bool open(const std::string &file_name)
    {
        if (file_name.size() <= 0)
        {
            QCERR("file name error");
            throw std::invalid_argument("file name error");
        }

        m_file_path = file_name;

#ifdef _MSC_VER
        using convert_typeX = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_typeX, wchar_t> converterX;

        auto w_file_path = converterX.from_bytes(m_file_path);

        std::wifstream ifs(w_file_path);
        if (!ifs)
        {
            QCERR("file error");
            //throw std::invalid_argument("file error");
            return false;
        }

        rapidjson::WIStreamWrapper isw(ifs);
        m_doc.ParseStream(isw);
        ifs.close();
#else
        std::ifstream ifs(file_name);
        if (!ifs)
        {
            QCERR("file error");
            //throw std::invalid_argument("file error");
            return false;
        }

        rapidjson::IStreamWrapper isw(ifs);
        m_doc.ParseStream(isw);
        ifs.close();
#endif  
        if (m_doc.HasParseError())
        {
            QCERR("Json pase error");
            //throw std::runtime_error("Json pase error");
            return false;
        }
        m_key_vector.resize(0);
        for (auto aiter = m_doc.MemberBegin();aiter != m_doc.MemberEnd();aiter++)
        {
            if ((*aiter).name.IsString())
                m_key_vector.push_back((*aiter).name.GetString());
            else
            {
                QCERR("Json name type error");
                //throw std::runtime_error("Json name type error");
                return false;
            }
        }
        
        return true;
        
    }

    /**
     * @brief write
     * Write json file
     * @return true  Write success
     * @return false Write fail
     */
    inline bool write()
    {
        if(m_file_path.size() <= 0)
        {
            return false;
        }

#ifdef _MSC_VER
        using convert_typeX = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_typeX, wchar_t> converterX;

        auto w_file_path = converterX.from_bytes(m_file_path);

        std::wofstream out_file_stream;
        out_file_stream.open(w_file_path, std::ios::ate);
        rapidjson::WOStreamWrapper out_stream_wapper(out_file_stream);
        rapidjson::Writer<rapidjson::WOStreamWrapper> write(out_stream_wapper);
        m_doc.Accept(write);
        out_file_stream.close();
#else
        std::ofstream out_file_stream;
        out_file_stream.open(m_file_path, std::ios::ate);
        rapidjson::OStreamWrapper out_stream_wapper(out_file_stream);
        rapidjson::Writer<rapidjson::OStreamWrapper> write(out_stream_wapper);
        m_doc.Accept(write);
        out_file_stream.close();
#endif // WIN32
        
        return true;
    }

    /**
     * @brief Get the Json String 
     * Get object json string
     * @return std::string 
     */
    std::string getJsonString()
    {
        std::string temp;
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> write(buffer);
        m_doc.Accept(write);
        temp.append(buffer.GetString());
        return temp;
    }

    /**
     * @brief Get the File Path 
     * 
     * @return std::string 
     */
    std::string getFilePath()
    {
        return m_file_path;
    }

    std::vector<std::string> getKeyVector()
    {
        return m_key_vector;
    }
};
QPANDA_END
#endif

