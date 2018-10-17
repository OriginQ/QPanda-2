/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "QuantumGates.h"
#include <time.h>

QuantumGates::QuantumGates()
{
}


QuantumGates::~QuantumGates()
{
}


/*****************************************************************************************************************
Name:        randGenerator
Description: 16807 random number generator
Argin:       None
Argout:      None
return:      random number in the region of [0,1]
*****************************************************************************************************************/
double QuantumGates::randGenerator()
{
    /*
     *  difine constant number in 16807 generator.
     */
    int  ia = 16807, im = 2147483647, iq = 127773, ir = 2836;
#ifdef _WIN32
    time_t rawtime;
    struct tm  timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    static int irandseed = timeinfo.tm_year + 70 *
                           (timeinfo.tm_mon + 1 + 12 *
                            (timeinfo.tm_mday + 31 *
                             (timeinfo.tm_hour + 23 *
                              (timeinfo.tm_min + 59 * timeinfo.tm_sec))));
#else
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo  = localtime(&rawtime);

    static int irandseed = timeinfo->tm_year + 70 *
                           (timeinfo->tm_mon + 1 + 12 *
                            (timeinfo->tm_mday + 31 *
                             (timeinfo->tm_hour + 23 *
                              (timeinfo->tm_min + 59 * timeinfo->tm_sec))));   
#endif
    static int irandnewseed = 0;
    if (ia * (irandseed % iq) - ir * (irandseed / iq) >= 0)
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq);
    }
    else
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq) + im;
    }
    irandseed = irandnewseed;
    return (double)irandnewseed / im;
}
