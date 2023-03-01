/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
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

/**
* @brief a wrapper class of rapidjson.
* @ingroup Utils
*/
class RJson
{
public:
	/**
	* @brief  No default Constructor of RJson
	*/
    RJson() = delete ;

	/**
	* @brief get the val by the name of a key
	* @param[out] rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address. 
	* @param[in] char* the name of key
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
    static int GetValue(
            const rapidjson::Value **value,
            const char *name,
            const rapidjson::Value *parent);

	/**
	* @brief get the val by index
	* @param[out] rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address. 
	* @param[in] size_t the target index
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
    static int GetValue(
            const rapidjson::Value **value,
            const size_t idx,
            const rapidjson::Value *parent);

	/**
	* @brief get the object by tag
	* @param[out] rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address. 
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the array by tag
	* @param[out] rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address.
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the string of tag
	* @param[out] std::string& the string of tag
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the bool val of tag
	* @param[out] bool& the bool val
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the int val of tag
	* @param[out] int& the int val
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the double val of tag
	* @param[out] double& the double val
	* @param[in] T tag
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the val of t1.t2
	* @param[out]  rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address.
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the object pointer of t1.t2
	* @param[out]  rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address.
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the Array pointer of t1.t2
	* @param[out]  rapidjson::Value** Pointer of a pointer, pointer to the storage Memory address.
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the string of t1.t2
	* @param[out]  std::string&  the target string
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the bool val of t1.t2
	* @param[out] bool&  the target bool val
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the int val of t1.t2
	* @param[out] int&  the target int val
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief get the double val of t1.t2
	* @param[out] double& the target double val
	* @param[in] T1 t1
	* @param[in] T2 t2
	* @param[in] rapidjson::Value* parent node pointer
	* @return int 0: on success, -1: others
	*/
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

	/**
	* @brief node data to string
	* @param[out] std::string& the string of node data
	* @param[in] rapidjson::Value* the target node
	* @return int 0: on success, -1: others
	*/
    static int ToString(
            std::string& str,
            const rapidjson::Value* node);

	/**
	* @brief node data to string
	* @param[in] rapidjson::Value* the target node
	* @return std::string the string of node data. if any error occurred, the returned string is empty.
	*/
    static std::string ToString(const rapidjson::Value* node);

	/**
	* @brief parse a faile
	* @param[in] std::string& the target file name
	* @param[in] rapidjson::Document& JSON file parser
	* @return bool return true on success, or else false on any error occurred
	*/
    static bool parse(const std::string &filename, rapidjson::Document &doc);

	/**
	* @brief judge the file whether valid or not.
	* @param[in] std::string& the target file name
	* @param[in] std::string& the config schema
	* @param[in] rapidjson::Document& JSON file parser
	* @return bool if file is valid return true, or else return false
	*/
    static bool validate(
            const std::string &filename,
            const std::string &config_schema,
            rapidjson::Document &doc
            );

};

}

#endif // RJSON_H
