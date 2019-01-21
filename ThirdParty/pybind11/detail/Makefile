SUBDIRS = $(shell ls -l | grep ^d | awk '{print $$9}')
CUR_SOURCE = ${wildcard *.cpp}
CUR_OBJS = ${patsubst %.cpp, %.o, $(CUR_SOURCE)}
all : $(SUBDIRS) $(CUR_OBJS)
$(SUBDIRS) : ECHO
    make -C $@
$(CUR_OBJS) : %.o:%.cpp
    $(CC) -c $^ -o $(ROOT_DIR)/$(AOBJS_DIR)/$@
ECHO:
    @echo $(SUBDIRS)
