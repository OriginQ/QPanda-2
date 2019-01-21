#include <fstream>
#include <iostream>
#include "RJson.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/schema.h"

namespace QPanda
{

int RJson::GetValue(
        const rapidjson::Value **value,
        const char *name,
        const rapidjson::Value *parent)
{
    if (parent && name && parent->IsObject())
        {
            rapidjson::Value::ConstMemberIterator
                    itr = parent->FindMember(name);
            if (itr != parent->MemberEnd())
            {
                *value = &(itr->value);
                return 0;
            }
        }

    return -1;
}

int RJson::GetValue(
        const rapidjson::Value **value,
        const size_t idx,
        const rapidjson::Value *parent)
{
    if (parent && parent->IsArray() && idx < parent->Size())
        {
            *value = &( (*parent)[(rapidjson::SizeType)idx]);
            return 0;
        }

    return -1;
}

int RJson::ToString(
        std::string &str,
        const rapidjson::Value *node)
{
    if (node)
    {
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        node->Accept(writer);
        str = sb.GetString();
        return 0;
    }

    return -1;
}

std::string RJson::ToString(const rapidjson::Value *node)
{
    if (node)
    {
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        node->Accept(writer);
        return sb.GetString();
    }

    return "";
}

bool RJson::parse(const std::string &filename, rapidjson::Document &doc)
{
    std::ifstream ifs(filename);
    rapidjson::IStreamWrapper isw_s(ifs);

    if (doc.ParseStream(isw_s).HasParseError())
    {
        std::cout << "The file content is not a valid JSON. "
                  << "filename: " << filename.c_str() << std::endl;

        ifs.close();
        return false;
    }

    ifs.close();
    return true;
}

bool RJson::validate(
        const std::string &filename,
        const std::string &config_schema,
        rapidjson::Document &doc)
{
    do
    {
        if (!parse(filename, doc))
        {
            break;
        }

        rapidjson::Document sd;
        if (sd.Parse<0>(config_schema.c_str()).HasParseError())
        {
            std::cout << "Schema content is not a valid JSON. " << std::endl;
            std::cout << config_schema.c_str() << std::endl;
            break;
        }

        rapidjson::SchemaDocument schema(sd);
        rapidjson::SchemaValidator validator(schema);
        if (!doc.Accept(validator)) {
            // Input JSON is invalid according to the schema
            // Output diagnostic information
            rapidjson::StringBuffer sb;
            validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
            std::cout << "Invalid schema: "
                      << sb.GetString() << std::endl;
            std::cout << "Invalid keyword: "
                      << validator.GetInvalidSchemaKeyword() << std::endl;
            sb.Clear();
            validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
            std::cout << "Invalid document: "
                      << sb.GetString() << std::endl;
        }

        return true;
    } while(0);

    return false;
}

}
