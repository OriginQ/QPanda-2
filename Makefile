CC = g++ -std=c++11 -g $(INCLUDE)
SUBDIRS = $(shell ls -l | grep ^d | awk '{if($$9 != "Release") print $$9}')
ROOT_DIR = $(shell pwd)
QBIN = libQPandaSDK.a
ABIN = libQPandaAlgorithm.a
BIN = test
AOBJS_DIR = Release/obj/Algorithm/
QOBJS_DIR = Release/obj/QPandaSDK/
COBJS_DIR = Release/obj/Console/
OBJS_DIR = Release/obj/
BIN_DIR = Release/bin
CONFIF_FILE = Config.xml
MEATADATA_FILE = MetadataConfig.xml
CUR_SOURCE = ${wildcard *.cpp}
CUR_OBJS = ${patsubst %.cpp, %.o, $(CUR_SOURCE)}
INCLUDE = -I $(ROOT_DIR)/Console \
		  -I $(ROOT_DIR)/QPanda-2.0.Algorithm \
		  -I $(ROOT_DIR)/QPanda-2.0.Algorithm/Algorithm \
		  -I $(ROOT_DIR)/QPanda-2.0.Windows/QPanda \
		  -I $(ROOT_DIR)/QPanda-2.0.Windows \
		  -I $(ROOT_DIR)/QPanda-2.0.Windows/QuantumInstructionHandle \
		  -I $(ROOT_DIR)/QPanda-2.0.Windows/QuantumMachin \
		  -I $(ROOT_DIR)/QPanda-2.0.Windows/TraversalAlgorithm \
		  -I $(ROOT_DIR)/TinyXML
export CC BIN ABIN QBIN OBJS_DIR AOBJS_DIR QOBJS_DIR COBJS_DIR BIN_DIR ROOT_DIR CONFIF_FILE MEATADATA_FILE
all : MKDIE $(SUBDIRS)  RELEASE
	
MKDIE :
	mkdir -p $(OBJS_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(AOBJS_DIR)
	mkdir -p $(QOBJS_DIR)
	mkdir -p $(COBJS_DIR)
$(SUBDIRS) : ECHO
	make -C $@
RELEASE : ECHO
	make -C Release
ECHO:
	@echo $(SUBDIRS)
clean:
	@rm -f $(COBJS_DIR)*.o
	@rm -f $(AOBJS_DIR)*.o
	@rm -f $(QOBJS_DIR)*.o
	@rm -rf $(BIN_DIR)/*
