#include "Core/Utilities/Utilities.h"
#include <time.h>
using namespace std;

USING_QPANDA
std::string QPanda::dec2bin(unsigned n, size_t size)
{
    std::string binstr = "";
    for (int i = 0; i < size; ++i)
    {
        binstr = (char)((n & 1) + '0') + binstr;
        n >>= 1;
    }
    return binstr;
}

double QPanda::RandomNumberGenerator()
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
    timeinfo = localtime(&rawtime);

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

void QPanda::add_up_a_map(map<string, size_t> &meas_result, string key)
{
    if (meas_result.find(key) != meas_result.end())
    {
        meas_result[key]++;
    }
    else
    {
        meas_result[key] = 1;
    }
}