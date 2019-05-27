/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

RJson.h

Author: LiYe
Created in 2018-11-09

Note:

The RJson is a wrapper class of rapidjson.

KeyWords Description
Object: json's {key: value} format.
Array : json's [e1, e2, e3] format.
Value : json's abstract data type, includes Object, Array, int, string
        and so on, is the base class of Object and Array.
GetStr/GetInt : get the part string value of a json data£¬returned by the 1st
        parameter, the last parameter is the parent node value, the middle
        parameters are used to locate the element's position(type is int) or
        assign the corresponding value of the key(type is const char*).

Usage Example:

using namespace rapidjson;
using std::string;

Document doc;
char jsonstr[] = "{ \"key1\": 123, \"key2\": \"string value\",
                    \"key3\": [100, 200, \"str300\"],
                    \"key4\": { \"obj1\": 1, \"obj2\": \"str42\"}}"

if (doc.Parse(jsonstr).HasParseError()) { ... }
// get key2 string value
std::string val2;
int result = Rjson::GetValue(val2, "key2", &doc);

// get key1 integer value
int val1;
int result = Rjson::GetValue(val1, "key1", &doc);

// get array [1] integer of key3
int val3_1;
int result = Rjson::GetValue(val3_1, "key3", 1, &doc);

// get array [2] string of key3
string val3_2;
int result = Rjson::GetValue(val3_2, "key3", 2, &doc);

// get object inter of key4
int val4_i;
int result = Rjson::GetValue(val4_i, "key4", "obj1", &doc);

// get object string of key3
string val4_s;
int result = Rjson::GetValue(val4_s, "key4", "obj2", &doc);

Refer to: value_he@csdn.net
*/

#ifndef RJSON_H
#define RJSON_H

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace QPanda
{

class RJson
{
public:
    RJson() = delete ;

    static int GetValue(
            const rapidjson::Value **value,
            const char *name,
            const rapidjson::Value *parent);

    static int GetValue(
            const rapidjson::Value **value,
            const size_t idx,
            const rapidjson::Value *parent);

    template<typename T>
    static int GetObject(
            const rapidjson::Value** value,
            T tag,
            const rapidjson::Value* parent)
    {
        if (0 == GetValue(value, tag, parent) && (*value)->IsObject())
        {
            return 0;
        }

        *value = nullptr;
        return -1;
    }

    template<typename T>
    static int GetArray(
            const rapidjson::Value** value,
            T tag,
            const rapidjson::Value* parent)
    {
        if (0 == GetValue(value, tag, parent) && (*value)->IsArray())
        {
            return 0;
        }

        *value = nullptr;
        return -1;
    }

    template<typename T>
    static int GetStr(
            std::string& str,
            T tag,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, tag, parent) && value->IsString())
        {
            str = value->GetString();
            return 0;
        }

        return -1;
    }

    template<typename T>
    static int GetBool(
            bool& n,
            T tag,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, tag, parent))
        {
            n = value->GetBool();
            return 0;
        }

        return -1;
    }

    template<typename T>
    static int GetInt(
            int& n,
            T tag,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, tag, parent))
        {
            n = value->GetInt();
            return 0;
        }

        return -1;
    }

    template<typename T>
    static int GetDouble(
            double& n,
            T tag,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, tag, parent))
        {
            n = value->GetDouble();
            return 0;
        }

        return -1;
    }

    template<typename T1, typename T2>
    static int GetValue(
            const rapidjson::Value** value,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* tmpv = nullptr;
        int ret = GetValue(&tmpv, t1, parent);
        if (0 == ret)
        {
            return GetValue(value, t2, tmpv);
        }

        return -1;
    }

    template<typename T1, typename T2>
    static int GetObject(
            const rapidjson::Value** value,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        if (0 == GetValue(value, t1, t2, parent) && (*value)->IsObject())
        {
            return 0;
        }
        *value = nullptr;
        return -1;
    }

    template<typename T1, typename T2>
    static int GetArray(
            const rapidjson::Value** value,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        if (0 == GetValue(value, t1, t2, parent) && (*value)->IsArray())
        {
            return 0;
        }
        *value = nullptr;
        return -1;
    }

    template<typename T1, typename T2>
    static int GetStr(
            std::string& str,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, t1, t2, parent) && value->IsString())
        {
            str = value->GetString();
            return 0;
        }

        return -1;
    }

    template<typename T1, typename T2>
    static int GetBool(
            bool& n,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, t1, t2, parent))
        {
            n = value->GetBool();
            return 0;
        }

        return -1;
    }

    template<typename T1, typename T2>
    static int GetInt(
            int& n,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, t1, t2, parent))
        {
            n = value->GetInt();
            return 0;
        }

        return -1;
    }

    template<typename T1, typename T2>
    static int GetDouble(
            double& n,
            T1 t1,
            T2 t2,
            const rapidjson::Value* parent)
    {
        const rapidjson::Value* value = nullptr;
        if (0 == GetValue(&value, t1, t2, parent))
        {
            n = value->GetDouble();
            return 0;
        }

        return -1;
    }

    static int ToString(
            std::string& str,
            const rapidjson::Value* node);

    static std::string ToString(const rapidjson::Value* node);

    static bool parse(const std::string &filename, rapidjson::Document &doc);

    static bool validate(
            const std::string &filename,
            const std::string &config_schema,
            rapidjson::Document &doc
            );

};

}

#endif // RJSON_H
