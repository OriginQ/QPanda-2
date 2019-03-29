#ifndef _GPU_STRUCT_H
#define _GPU_STRUCT_H

#define QSIZE   size_t
#ifndef SQ2
#define SQ2 (1 / 1.4142135623731)
#endif

#ifndef PI
#define PI 3.14159265358979
#endif

#define THREADDIM (1024)
#define STATE_T double

namespace GATEGPU
{
    struct probability
    {
        STATE_T prob;
        int state;
    };

    struct QState
    {
        QState() : real(nullptr), imag(nullptr) {}
        STATE_T * real;
        STATE_T * imag;
        size_t qnum;
    };
}
#endif // !_GPU_STRUCT_H



