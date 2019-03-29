#ifndef  _SINGLEAMPLITUDE_H_
#define  _SINGLEAMPLITUDE_H_
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
#include "include/Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
QPANDA_BEGIN

class SingleAmplitudeQVM : public QVM, public TraversalQProg
{
public:
    SingleAmplitudeQVM();
    ~SingleAmplitudeQVM() {};

   void traversalAll(AbstractQuantumProgram*);
   void traversal(AbstractQGateNode*);

   void run(QProg&);
   QStat getQStat();

   double PMeasure_index(size_t);
   vector<double> PMeasure(QVec, int);
   std::vector<std::pair<size_t, double>> PMeasure(int);

   std::vector<double> getProbList(QVec, int);
   std::vector<double> probRunList(QProg &, QVec, int);

   std::map<std::string, double> getProbDict(QVec, int);
   std::map<std::string, double> probRunDict(QProg &, QVec, int);

   std::vector<std::pair<size_t, double>> getProbTupleList(QVec, int);
   std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec, int);

private:
    QuantumProgMap m_prog_map;

    std::map<size_t, std::function<void(QuantumProgMap &, size_t, bool)> > m_singleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, bool)>> m_doubleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, double, bool)>> m_singleAngleGateFunc;
    std::map<size_t, std::function<void(QuantumProgMap &, size_t, size_t, double, bool)>> m_doubleAngleGateFunc;
};

QPANDA_END
#endif
