/// CMemFactory test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(CMemFactoryInfaceTest, test) {

	CMemFactory memFac = CMemFactory::GetFactoryInstance();
	CMem* mem = memFac.GetInstanceFromSize(8);
	EXPECT_EQ(8, mem->getMaxMem());
	mem->clearAll();
	
}