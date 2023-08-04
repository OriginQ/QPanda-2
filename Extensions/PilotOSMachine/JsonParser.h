#ifndef PILOTOS_JSON_PARSER_H
#define PILOTOS_JSON_PARSER_H

#include "QPandaConfig.h"
#include <set>
#include <vector>
#include <stdexcept>
#include "ThirdParty/rapidjson/rapidjson.h"
#include "Def.h"

namespace JsonMsg 
{
    class JsonParser
    {
#define _GET_MEMBER(_key_str, type) type _ret_val;_get_member(_key_str, _ret_val);return _ret_val;
    public:
        JsonParser()
            :m_b_load_json_failed(true)
        {}

        explicit JsonParser(rapidjson::Value& json_val)
            :m_json_obj(std::move(json_val)), m_b_load_json_failed(true)
        {}

        explicit JsonParser(const std::string& json_str)
            :m_b_load_json_failed(true)
        {
            load_json(json_str.c_str());
        }

        JsonParser(JsonParser&& _other)
            :m_json_obj(std::move(_other.m_json_obj))
        {}

        ~JsonParser() {}

        JsonParser operator[](const std::string& key)
        {
            if (m_json_obj.HasMember(key.c_str())){
                return JsonParser(m_json_obj[key.c_str()]);
            }
            return {};
        }

        bool load_json(const std::string &str_json, const std::string& msg_type = ""){
            return load_json(str_json.c_str(), msg_type);
        }

        /* Load the JSON string. If the parsing is wrong, return false; otherwise, return true */
        bool load_json(const char* json_str, const std::string& msg_type = "") 
        {
            if (m_doc.Parse(json_str).HasParseError())
            {
                return false;
            }

            m_json_obj = m_doc.Move();
            if (!msg_type.empty())
            {
                if (m_json_obj.HasMember(MSG_TYPE) && m_json_obj[MSG_TYPE].IsString())
                {
                    if (msg_type != m_json_obj[MSG_TYPE].GetString()) {
                        return false;
                    }
                }
            }
            m_b_load_json_failed = false;
            return true;
        }

        bool has_member(const std::string& key) {
            return m_json_obj.IsObject() && m_json_obj.HasMember(key.c_str());
        }

        /* check whether has bool member: \p key */
        bool has_member_bool(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsBool();
        }

        /* check whether has int32_t member: \p key */
        bool has_member_int32(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsInt();
        }

        /* check whether has uint32_t member: \p key */
        bool has_member_uint32(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].GetUint();
        }

        /* check whether has float member: \p key */
        bool has_member_float(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsFloat();
        }

        /* check whether has double member: \p key */
        bool has_member_double(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsDouble();
        }

        /* check whether has array member: \p key */
        bool has_member_array(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsArray();
        }

        /* check whether has string member: \p key */
        bool has_member_string(const std::string& key) {
            return has_member(key.c_str()) && m_json_obj[key.c_str()].IsString();
        }

        rapidjson::Value& get_json_obj() { return m_json_obj; }
        rapidjson::Value& get_json_obj(const std::string& key) { return m_json_obj[key.c_str()]; }

        uint32_t get_uint32(const std::string& key){
            _GET_MEMBER(key.c_str(), uint32_t);
        }

        uint64_t get_uint64(const std::string& key){ 
            _GET_MEMBER(key.c_str(), uint64_t); 
        }

        double get_double(const std::string& key){
            _GET_MEMBER(key.c_str(), double);
        }

        float get_float(const std::string& key){
            _GET_MEMBER(key.c_str(), float);
        }

        std::string get_string(const std::string& key) {
            _GET_MEMBER(key.c_str(), std::string);
        }

        int32_t get_int32(const std::string& key) {
            _GET_MEMBER(key.c_str(), int32_t);
        }

        bool get_bool(const std::string& key){
            _GET_MEMBER(key.c_str(), bool);
        }

        template <class Ty>
        void get_set(const char* key, std::set<Ty>& vec) {
            vec.clear();
            if ((m_json_obj.HasMember(key)) && m_json_obj[key].IsArray()) {
                rapidjson::Value& arr_data = m_json_obj[key];
                for (int i = 0; i < arr_data.Size(); ++i) {
                    vec.insert(_get_val(arr_data[i], Ty()));
                }
            }
        }

        template <class Ty>
        bool get_array_2d(const std::string& key, std::vector<std::vector<Ty>>& array_2D){
            try
            {
                _get_array_2d(key.c_str(), array_2D);
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        std::vector<std::vector<uint32_t>> get_array_2d(const std::string& key)
        {
            std::vector<std::vector<uint32_t>> _array_2D;
            _get_array_2D(key.c_str(), _array_2D);
            return _array_2D;
        }

        template <class Ty1, class Ty2>
        bool get_dict(const std::string& key, std::map<Ty1, Ty2>& _map) 
        {
            _map.clear();
            if (m_json_obj.HasMember(key.c_str())) {
                std::vector<Ty1> key_vec;
                _get_array_(m_json_obj[key.c_str()], MSG_DICT_KEY, key_vec);
                std::vector<Ty2> val_vec;
                _get_array_(m_json_obj[key.c_str()], MSG_DICT_VALUE, val_vec);
                if (key_vec.size() != val_vec.size()) {
                    throw std::runtime_error("json parse fail:key error " + (std::string)key);
                }
                for (size_t i = 0; i < key_vec.size(); ++i) {
                    _map.insert(std::make_pair(key_vec[i], val_vec[i]));
                }
            }
            else {
                return false;
            }
            return true;
        }

        template <class Ty>
        void get_array_2d(const char* key, std::vector<std::set<Ty>>& mat) {
            mat.clear();
            if ((m_json_obj.HasMember(key)) && m_json_obj[key].IsArray()) {
                rapidjson::Value& arr_row = m_json_obj[key];
                mat.resize(arr_row.Size());
                for (int i = 0; i < arr_row.Size(); ++i) {
                    auto& _row_data = mat[i];
                    rapidjson::Value& arr_col = arr_row[i];
                    for (int j = 0; j < arr_col.Size(); ++j) {
                        _row_data.insert(_get_val(arr_col[j], Ty()));
                    }
                }
            }
        }

        template <class Ty>
        void get_array(const char* key, std::vector<Ty>& vec) {
            vec.clear();
            if ((m_json_obj.HasMember(key)) && m_json_obj[key].IsArray()) {
                rapidjson::Value& arr_data = m_json_obj[key];
                for (int i = 0; i < arr_data.Size(); ++i) {
                    vec.emplace_back(_get_val(arr_data[i], Ty()));
                }
            }
        }

        std::vector<std::string> get_array(const std::string& key)
        {
            std::vector<std::string> str_vec;
            get_array(key.c_str(), str_vec);
            return str_vec;
        }

    protected:
        template <class Ty>
        bool _check_val(rapidjson::Value& json_val, Ty _default_val) { return false; }

        inline bool _check_val(rapidjson::Value& json_val, uint32_t __) {
            return json_val.IsUint();
        }

        inline bool _check_val(rapidjson::Value& json_val, int32_t __) {
            return json_val.IsInt();
        }

        inline bool _check_val(rapidjson::Value& json_val, size_t __) {
            return json_val.IsUint64();
        }

        inline bool _check_val(rapidjson::Value& json_val, std::string __) {
            return json_val.IsString();
        }

        inline bool _check_val(rapidjson::Value& json_val, double __) {
            return json_val.IsDouble();
        }

        inline bool _check_val(rapidjson::Value& json_val, bool __) {
            return json_val.IsBool();
        }

        template <class Ty>
        Ty _get_val(rapidjson::Value& json_val, Ty _default_val) { return _default_val; }

        inline uint32_t _get_val(rapidjson::Value& json_val, uint32_t __) {
            return json_val.GetUint();
        }

        inline int _get_val(rapidjson::Value& json_val, int32_t __) {
            return json_val.GetInt();
        }

        inline size_t _get_val(rapidjson::Value& json_val, size_t __) {
            return json_val.GetUint64();
        }

        inline std::string _get_val(rapidjson::Value& json_val, std::string __) {
            return json_val.GetString();
        }

        inline double _get_val(rapidjson::Value& json_val, double __) {
            return json_val.GetDouble();
        }

        inline bool _get_val(rapidjson::Value& json_val, bool __) {
            return json_val.GetBool();
        }

        template <class Ty>
        void _get_array_2D(const char* key, std::vector<std::vector<Ty>>& array_2D) {
            _get_matrix(key, array_2D);
        }

        template <class Ty>
        void _get_matrix(const char* key, std::vector<std::vector<Ty>>& mat) {
            mat.clear();
            if ((m_json_obj.HasMember(key)) && m_json_obj[key].IsArray()) {
                rapidjson::Value& arr_row = m_json_obj[key];
                mat.resize(arr_row.Size());
                for (int i = 0; i < arr_row.Size(); ++i) {
                    auto& _row_data = mat[i];
                    rapidjson::Value& arr_col = arr_row[i];
                    for (int j = 0; j < arr_col.Size(); ++j) {
                        _row_data.push_back(_get_val(arr_col[j], Ty()));
                    }
                }
            }
        }
        
        template <class Ty>
        void _get_member(const char* key, Ty& _member_val) {
            _member_val = {};
            if (m_json_obj.HasMember(key) && (_check_val(m_json_obj[key], _member_val))) 
            {
                _member_val = _get_val(m_json_obj[key], _member_val);
            }
        }
        
        template <class Ty>
        void _get_array_(rapidjson::Value& json_val, const char* key, std::vector<Ty>& vec) {
            vec.clear();
            if ((json_val.HasMember(key)) && json_val[key].IsArray()) {
                rapidjson::Value& arr_data = json_val[key];
                for (int i = 0; i < arr_data.Size(); ++i) 
                {
                    vec.emplace_back(_get_val(arr_data[i], Ty()));
                }
            }
        }

        template <class Ty>
        void _get_matrix_(rapidjson::Value& json_val, const char* key, std::vector<std::vector<Ty>>& mat) {
            mat.clear();
            if ((json_val.HasMember(key)) && json_val[key].IsArray()) {
                rapidjson::Value& arr_row = json_val[key];
                mat.resize(arr_row.Size());
                for (int i = 0; i < arr_row.Size(); ++i) {
                    auto& _row_data = mat[i];
                    rapidjson::Value& arr_col = arr_row[i];
                    for (int j = 0; j < arr_col.Size(); ++j) {
                        _row_data.push_back(_get_val(arr_col[j], Ty()));
                    }
                }
            }
        }

    private:
        rapidjson::Document m_doc;
        rapidjson::Value m_json_obj;
        bool m_b_load_json_failed;         /**< Parse error flag */
    };

    inline std::string object_to_string(rapidjson::Value& val_obj)
    {
        if (!val_obj.IsObject() && !val_obj.IsArray()) 
        {
            return {};
        }
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        val_obj.Accept(writer);
        return std::string(buf.GetString());
    }
}

#endif

