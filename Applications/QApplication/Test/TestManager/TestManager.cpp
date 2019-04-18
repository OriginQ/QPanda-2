#include <string>
#include <iostream>
#include "TestManager.h"
#include "TestInterface/AbstractTest.h"
#include "RJson/RJson.h"
#include "config_schema.h"

namespace QPanda
{

    SINGLETON_IMPLEMENT_LAZY(TestManager);

    TestManager::TestManager():
        m_config_schema(kConfigSchema)
    {

    }

    bool TestManager::exec()
    {
        rapidjson::Document doc;

        if (!RJson::validate(m_file, m_config_schema, doc))
        {
            return false;
        }

        std::string test_alg_name;
        RJson::GetStr(test_alg_name, "algorithm", "name", &doc);

        for (size_t i = 0; i < m_test_vec.size(); i++)
        {
            auto test = m_test_vec[i];
            if (test->canHandle(test_alg_name))
            {
                return test->exec(doc);
            }
        }

        return false;
    }

}
