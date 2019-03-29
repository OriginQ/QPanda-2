#ifndef  _PARTIALAMPLITUDE_H_
#define  _PARTIALAMPLITUDE_H_
#include "include/Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/MergeMap.h"
QPANDA_BEGIN

class PartialAmplitudeQVM : public QVM
{
public:
    PartialAmplitudeQVM();
    ~PartialAmplitudeQVM();

    void run(QProg&);
    QStat getQStat();

    std::vector<double> PMeasure(QVec, int);
    std::vector<std::pair<size_t, double> > PMeasure(int);

    std::vector<double> getProbList(QVec, int);
    std::vector<double> probRunList(QProg &, QVec, int);

    std::map<std::string, double> getProbDict(QVec, int);
    std::map<std::string, double> probRunDict(QProg &, QVec, int);

    std::vector<std::pair<size_t, double>> getProbTupleList(QVec, int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec, int);

private:
    MergeMap *m_prog_map;
    long long low_pos, high_pos;

    void getAvgBinary(long long, size_t);
};

QPANDA_END
#endif
