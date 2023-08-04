#pragma once

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include <map>
#include <vector>
#include <string>
#include "Def.h"
#include "rabbit.hpp"


namespace JsonMsg {
    class JsonBuilder
    {
    public:
        JsonBuilder() { m_doc.Parse("{}"); };
        ~JsonBuilder() {};

        rapidjson::Document& get_json_obj() { return m_doc; }

        void set_msg_type(const std::string& msg_type) {
            add_member(MSG_TYPE, msg_type);
        }

        void add_member(const std::string& key, const std::string& val) {
            m_doc.AddMember(build_string_val(key), build_string_val(val), m_doc.GetAllocator());
        }

        void add_member(const std::string& key, const uint32_t & val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.SetUint(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        bool has_member(const std::string& key) {
            if (!m_doc.IsObject()) {
                return false;
            }
            return m_doc.HasMember(key.c_str());
        }

        void add_member(const std::string& key, const uint64_t& val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.SetUint64(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_member(const std::string& key, const double& val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.SetDouble(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_member(const std::string& key, const int& val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.SetInt(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_member(const std::string& key, const long long& val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.SetInt64(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_member_bool(const std::string& key, bool val) {
            rapidjson::Value val_;
            val_.SetBool(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_member(const std::string& key, const rapidjson::Value& val) {
            auto &allocator = m_doc.GetAllocator();
            rapidjson::Value _val;
            _val.CopyFrom(val, m_doc.GetAllocator());
            m_doc.AddMember(build_string_val(key), _val, allocator);
        }

        void update_member(const std::string& key, rapidjson::Value& val) {
            m_doc.RemoveMember(key.c_str());

            auto &allocator = m_doc.GetAllocator();
            rapidjson::Value _val;
            _val.CopyFrom(val, m_doc.GetAllocator());
            m_doc.AddMember(build_string_val(key), _val, allocator);
        }

        void remove_member(const std::string& key) {
            if (m_doc.HasMember(key.c_str())) {
                m_doc.RemoveMember(key.c_str());
            }
        }

        void remove_member(rapidjson::Value &value, const std::string& key)
        {
            if (value.HasMember(key.data()))
            {
                value.RemoveMember(key.data());
            }
        }

        void add_member(const std::string& key, rapidjson::Document&& doc) {
            auto& allocator = m_doc.GetAllocator();
            rapidjson::Value _val;
            _val.CopyFrom(doc, m_doc.GetAllocator());
            m_doc.AddMember(build_string_val(key), _val, allocator);
        }

        template<class T>
        void add_member_to_obj(rapidjson::Value& obj, const std::string& key, const T& val) {
            rapidjson::Value val_;
            val_.Set(val);
            obj.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_string_to_obj(rapidjson::Value& obj, const std::string& key, const std::string& val) {
            obj.AddMember(build_string_val(key), build_string_val(val), m_doc.GetAllocator());
        }

        template<class T>
        void add_number_member(const std::string& key, const T& val) {
            rapidjson::Value val_(rapidjson::kNumberType);
            val_.Set(val);
            m_doc.AddMember(build_string_val(key), val_, m_doc.GetAllocator());
        }

        void add_array(const std::string& name, const std::vector<std::string> &arry) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto &_allocator = m_doc.GetAllocator();

            for (const auto &item : arry) {
                arr_val.PushBack(build_string_val(item), _allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, _allocator);
            return;
        }

        void add_array(const std::string& name, const std::vector<uint32_t> &arry) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto &allocator = m_doc.GetAllocator();

            for (const auto &item : arry)
            {
                rapidjson::Value int_key(rapidjson::kNumberType);
                int_key.SetInt(item);
                arr_val.PushBack(int_key, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
            return;
        }

        void add_array(const std::string& name, const std::vector<uint64_t> &arry) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto &allocator = m_doc.GetAllocator();

            for (const auto &item : arry)
            {
                rapidjson::Value int_key(rapidjson::kNumberType);
                int_key.SetUint64(item);
                arr_val.PushBack(int_key, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
            return;
        }

        void add_set(const std::string& name, const std::set<uint32_t>& set) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto& allocator = m_doc.GetAllocator();

            for (const auto& item : set)
            {
                rapidjson::Value int_key(rapidjson::kNumberType);
                int_key.SetInt(item);
                arr_val.PushBack(int_key, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
            return;
        }

        void add_array(const std::string& name, const std::vector<double>& arry) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto& allocator = m_doc.GetAllocator();

            for (const auto& item : arry)
            {
                rapidjson::Value double_val(rapidjson::kNumberType);
                double_val.SetDouble(item);
                arr_val.PushBack(double_val, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
            return;
        }

        /*void add_array(const std::string& name, const std::vector<std::vector<uint32_t>>& arry_2D) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto& allocator = m_doc.GetAllocator();

            for (const auto& sub_array : arry_2D)
            {
                rapidjson::Value sub_array_val(rapidjson::kArrayType);
                for (const auto &data_item : sub_array)
                {
                    rapidjson::Value int_key(rapidjson::kNumberType);
                    int_key.SetUint(item);
                    sub_array_val.PushBack(int_key, allocator);
                }

                arr_val.PushBack(sub_array_val, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
            return;
        }*/
        template <class Ty>
        void add_array(const std::string& name, const std::vector<std::set<Ty>>& arry_2D)
        {
            auto& allocator = m_doc.GetAllocator();
            rabbit::array _2d_array_json(allocator);
            for (const auto& sub_array : arry_2D)
            {
                rabbit::array _sub_array_json(allocator);
                for (const auto& data_item : sub_array) {
                    _sub_array_json.push_back(data_item);
                }

                _2d_array_json.push_back(_sub_array_json);
            }

            auto p_val = _2d_array_json.get_native_value_pointer();
            m_doc.AddMember(build_string_val(name), *p_val, allocator);
            return;
        }


        template <class Ty>
        void add_array(const std::string& name, const std::vector<std::vector<Ty>>& arry_2D)
        {
            auto& allocator = m_doc.GetAllocator();
            rabbit::array _2d_array_json(allocator);
            for (const auto& sub_array : arry_2D)
            {
                rabbit::array _sub_array_json(allocator);
                for (const auto &data_item : sub_array) {
                    _sub_array_json.push_back(data_item);
                }

                _2d_array_json.push_back(_sub_array_json);
            }

            auto p_val = _2d_array_json.get_native_value_pointer();
            m_doc.AddMember(build_string_val(name), *p_val, allocator);
            return;
        }

        void add_array(const std::string& name, const std::string& key1, const std::string& key2,
            const std::vector<std::pair<int, std::string>>& arry) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto &allocator = m_doc.GetAllocator();

            for (const auto &item : arry)
            {
                rapidjson::Value v1(rapidjson::kNumberType);
                v1.SetInt(item.first);

                rapidjson::Value dict(rapidjson::kObjectType);
                dict.AddMember(build_string_val(key1), v1, allocator);
                dict.AddMember(build_string_val(key2), build_string_val(item.second), allocator);
                arr_val.PushBack(dict, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
        }

#define ELSE_IF_FUN(type, fun) \
        else if(checkTupleKIndexType<type,TupleV>(i)) { \
            rapidjson::Value v(rapidjson::kNumberType);\
            v.Set##fun((type)convertType<type>(singlev));\
            dict.AddMember(build_string_val(ks[i]), v, allocator);\
        }\

        template<typename TupleK, typename TupleV, int N = std::tuple_size<TupleK>::value>
        void add_array(const std::string& name, TupleK keys, const std::vector<TupleV>& values) {
            rapidjson::Value arr_val;
            arr_val.SetArray();
            auto &allocator = m_doc.GetAllocator();
            std::vector<std::string> ks;

            travel_tuple(keys, [&](std::size_t i, auto&& singlev) {
                auto data = std::string(singlev);
                ks.push_back(data);
                });

            for (const auto &item : values)
            {
                rapidjson::Value dict(rapidjson::kObjectType);

                travel_tuple(item, [&](std::size_t i, auto&& singlev) {
                    if (checkTupleKIndexType<int, TupleV>(i)) {
                        rapidjson::Value v(rapidjson::kNumberType);
                        v.SetInt((int)convertType<int>(singlev));
                        dict.AddMember(build_string_val(ks[i]), v, allocator);
                    }
                    ELSE_IF_FUN(unsigned int, Uint)
                        ELSE_IF_FUN(short, Int)
                        ELSE_IF_FUN(unsigned short, Uint)
                        ELSE_IF_FUN(long long, Int64)
                        ELSE_IF_FUN(uint64_t, Uint64)
                        ELSE_IF_FUN(float, Float)
                        ELSE_IF_FUN(double, Double)
                    else if (checkTupleKIndexType<bool, TupleV>(i)) {
                        bool istrue = convertType<bool>(singlev);
                        if (istrue) {
                            rapidjson::Value v(rapidjson::kTrueType);
                            v.SetBool(istrue);
                            dict.AddMember(build_string_val(ks[i]), v, allocator);
                        }
                        else {
                            rapidjson::Value v(rapidjson::kFalseType);
                            v.SetBool(istrue);
                            dict.AddMember(build_string_val(ks[i]), v, allocator);
                        }
                    }
                    else if (checkTupleKIndexType<std::string, TupleV>(i)) {
                        rapidjson::Value v = build_string_val(convertType<std::string>(singlev));
                        dict.AddMember(build_string_val(ks[i]), v, allocator);
                    }
                    else if (checkTupleKIndexType<const char*, TupleV>(i)) {
                        rapidjson::Value v = build_string_val(std::string(convertType<const char*>(singlev)));
                        dict.AddMember(build_string_val(ks[i]), v, allocator);
                    }
                    else {
                        QCERR_AND_THROW(invalid_argument, "Error: add_array error, unknow type");
                    }
                    });

                arr_val.PushBack(dict, allocator);
            }

            m_doc.AddMember(build_string_val(name), arr_val, allocator);
        }

        void add_dict(const std::string& name, const std::map<std::string, double> &dict) {
            //rapidjson::Value arr_val(rapidjson::kArrayType);
            //arr_doc.SetArray();
            auto &allocator = m_doc.GetAllocator();

            rapidjson::Value value_res(rapidjson::kObjectType);
            rapidjson::Value key_array(rapidjson::kArrayType);
            rapidjson::Value value_array(rapidjson::kArrayType);

            for (const auto &item : dict)
            {
                key_array.PushBack(build_string_val(item.first), allocator);
                value_array.PushBack(item.second, allocator);
            }

            value_res.AddMember(MSG_DICT_KEY, key_array, allocator);
            value_res.AddMember(MSG_DICT_VALUE, value_array, allocator);

            //arr_val.PushBack(value_res, allocator);
            m_doc.AddMember(build_string_val(name), value_res, allocator);
            return;
        }

        void add_dict(const std::string& name, const std::map<std::string, std::string>& dict) {
            auto& allocator = m_doc.GetAllocator();

            rapidjson::Value value_res(rapidjson::kObjectType);
            rapidjson::Value key_array(rapidjson::kArrayType);
            rapidjson::Value value_array(rapidjson::kArrayType);

            for (const auto& item : dict)
            {
                key_array.PushBack(build_string_val(item.first), allocator);
                value_array.PushBack(build_string_val(item.second), allocator);
            }

            value_res.AddMember(MSG_DICT_KEY, key_array, allocator);
            value_res.AddMember(MSG_DICT_VALUE, value_array, allocator);

            m_doc.AddMember(build_string_val(name), value_res, allocator);
            return;
        }

        void add_dict(const std::map<std::string, double>& dict) {
            auto& allocator = m_doc.GetAllocator();
            rapidjson::Value key_array(rapidjson::kArrayType);
            rapidjson::Value value_array(rapidjson::kArrayType);

            for (const auto& item : dict)
            {
                key_array.PushBack(build_string_val(item.first), allocator);
                value_array.PushBack(item.second, allocator);
            }

            m_doc.AddMember(MSG_DICT_KEY, key_array, allocator);
            m_doc.AddMember(MSG_DICT_VALUE, value_array, allocator);
            return;
        }

        std::string get_json_str(bool b_pretty = false) {
            rapidjson::StringBuffer buffer;

            if (b_pretty) {
                rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
                m_doc.Accept(writer);
            }
            else {
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                m_doc.Accept(writer);
            }

            return std::string(buffer.GetString());
        }

        rapidjson::Document& get_json_doc()
        {
            return m_doc;
        }

    protected:
        inline rapidjson::Value build_string_val(const std::string& str) {
            rapidjson::Value string_val(rapidjson::kStringType);
            string_val.SetString(str.c_str(), (rapidjson::SizeType)str.size(), m_doc.GetAllocator());
            return string_val;
        }


        template<std::size_t K, typename Type, typename Tuple, std::size_t... Index>
        bool checkTupleKIndexType_impl(std::size_t _k, std::index_sequence<Index...>)
        {
            constexpr auto nums = std::array<bool, K>{std::is_same<std::tuple_element_t<Index, Tuple>, Type>::value ...};
            if (_k < K) {
                return nums[_k];
            }
            return false;
        }

        template<typename Type, typename Tuple>
        bool checkTupleKIndexType(std::size_t k)
        {
            constexpr auto size = std::tuple_size<typename std::decay<Tuple>::type>::value;
            return checkTupleKIndexType_impl<size, Type, Tuple>(k, std::make_index_sequence<size>{});
        }

        template <typename Tuple, typename Func, size_t ... N>
        void func_call_tuple(const Tuple& t, Func&& func, std::index_sequence<N...>) {
            static_cast<void>(std::initializer_list<int>{(func(N, std::get<N>(t)), 0)...});
        }

        template <typename ... Args, typename Func>
        void travel_tuple(const std::tuple<Args...>& t, Func&& func) {
            func_call_tuple(t, std::forward<Func>(func), std::make_index_sequence<sizeof...(Args)>{});
        }

        template <typename T, typename R>
        const T& convertType(const R& r) {
            return *reinterpret_cast<const T*>(&r);
        }

    private:
        rapidjson::Document m_doc;
    };

}


