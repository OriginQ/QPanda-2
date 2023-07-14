
#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA


TEST(CBitFactoryInfaceTest, test) {
	CBitFactory cbitFac =  CBitFactory::GetFactoryInstance();
	auto cbit = cbitFac.CreateCBitFromName("c1");
	EXPECT_STREQ("c1", cbit->getName().c_str());

	cbit->setOccupancy(true);
	EXPECT_TRUE(cbit->getOccupancy());

	EXPECT_EQ(1,cbit->get_addr());

	
}
