#include <time.h>
#include "AbstractTest.h"
#include "Optimizer/AbstractOptimizer.h"

namespace QPanda
{

AbstractTest::AbstractTest(const std::string &tag):
    m_tag(tag)
{

}

AbstractTest::~AbstractTest()
{

}

void AbstractTest::setOptimizerPara(
        AbstractOptimizer *optimizer,
        rapidjson::Value &value)
{
    if (nullptr == optimizer)
    {
        return;
    }

    if (value.HasMember(STR_MAX_ITER))
    {
        optimizer->setMaxIter(
                    static_cast<size_t>(value[STR_MAX_ITER].GetInt()));
    }

    if (value.HasMember(STR_MAX_FCALLS))
    {
        optimizer->setMaxFCalls(
                    static_cast<size_t>(value[STR_MAX_FCALLS].GetInt()));
    }

    if (value.HasMember(STR_XATOL))
    {
        optimizer->setXatol(value[STR_XATOL].GetDouble());
    }

    if (value.HasMember(STR_FATOL))
    {
        optimizer->setFatol(value[STR_FATOL].GetDouble());
    }

    if (value.HasMember(STR_DISP))
    {
        optimizer->setDisp(value[STR_DISP].GetBool());
    }

    if (value.HasMember(STR_ADAPTVE))
    {
        optimizer->setAdaptive(value[STR_ADAPTVE].GetBool());
    }
}

std::string AbstractTest::getOutputFile(
        rapidjson::Value &value,
        std::string default_name_prefix)
{
    value.IsString();
    if (value.HasMember(STR_OUTPUT_FILE))
    {
        std::string output_file;
        RJson::GetStr(output_file, STR_OUTPUT_FILE, &value);

        return output_file;
    }
    else
    {
        time_t now = time(nullptr);
        tm *ltm = localtime(&now);
        auto year = 1900 + ltm->tm_year;
        auto month = 1 + ltm->tm_mon;
        auto day = ltm->tm_mday;
        auto hour = ltm->tm_hour;
        auto min = ltm->tm_min;
        auto sec = ltm->tm_sec;

        char tmp_str[20];
        sprintf(tmp_str,
                "%04d%02d%02d_%02d%02d%02d",
                year, month, day,hour, min, sec);

        std::string filename = default_name_prefix +
                std::string(tmp_str) + ".txt";

        return filename;
    }
}

}
