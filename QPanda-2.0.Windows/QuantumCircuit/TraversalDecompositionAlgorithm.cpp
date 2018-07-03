#include "TraversalDecompositionAlgorithm.h"
#include "QPanda.h"


TraversalDecompositionAlgorithm::TraversalDecompositionAlgorithm()
{
}


TraversalDecompositionAlgorithm::~TraversalDecompositionAlgorithm()
{
}

void TraversalDecompositionAlgorithm::CUToControlSingleGate(AbstractQGateNode * pNode)
{
    if (pNode->getQuBitNum() == 1)
    {
        return;
    }

   auto pControlQuBit = pNode->popBackQuBit();
   vector<Qubit *> contorlQubitVector = { pControlQuBit };
   pNode->setControl(contorlQubitVector);
   auto pQGate = pNode->getQGate();

   if (nullptr == pQGate)
   {
       throw exception();
   }

   vector<Qubit*> qubitVector;
   if (pNode->getQuBitVector(qubitVector) <= 0)
   {
       throw exception();
   }
   auto targetQubit = qubitVector[0];
   auto pU4 = new U4(pQGate->getAlpha(), pQGate->getBeta(), pQGate->getGamma(),pQGate->getDelta());
   delete(pQGate);
   pNode->setQGate(pU4);

}
