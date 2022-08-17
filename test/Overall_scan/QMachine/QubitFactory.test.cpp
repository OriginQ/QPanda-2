
#include "QPanda.h"
#include "gtest/gtest.h"


TEST(QubitFactoryInfaceTest,test) {
	QubitFactory qbitfac = QubitFactory::GetFactoryInstance();

	PhysicalQubit* py = PhysicalQubitFactory::GetFactoryInstance().GetInstance();
	py->setQubitAddr(0);
	py->setOccupancy(true);
	auto qbit = qbitfac.GetInstance(py);
	EXPECT_TRUE(qbit->getOccupancy());
	EXPECT_EQ(0,qbit->get_phy_addr());

}