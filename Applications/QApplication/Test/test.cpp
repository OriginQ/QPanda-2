#include <iostream>
#include <cstdio>
#include "TestManager/TestManager.h"

int main(int argc, char* argv[])
{
    auto instance = QPanda::TestManager::getInstance();

    if (2 != argc)
    {
        std::cout  << "Please input the config file: ";
        std::string config_fileanme;
        std::cin >> config_fileanme;

        instance->setConfigFile(config_fileanme);
    }
    else
    {
        instance->setConfigFile(argv[1]);
    }

    if (!instance->exec())
    {
        std::cout << "Do test failed." << std::endl;
    }
    else
    {
        std::cout << "Do test success." << std::endl;
    }

    return 0;
}
