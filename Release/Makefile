OBJS = *.o
ODIR = obj
OUT  = $(ROOT_DIR)/$(BIN_DIR)/$(BIN)
ALIB = $(ROOT_DIR)/$(BIN_DIR)/$(ABIN)
QLIB = $(ROOT_DIR)/$(BIN_DIR)/$(QBIN)
all : $(OUT) $(ALIB) $(QLIB) 
	mkdir -p $(ROOT_DIR)/$(BIN_DIR)/ConfigFile
	cp $(ROOT_DIR)/Release/$(CONFIF_FILE) $(ROOT_DIR)/$(BIN_DIR)/ConfigFile/
	cp $(ROOT_DIR)/Release/$(MEATADATA_FILE) $(ROOT_DIR)/$(BIN_DIR)/ConfigFile/
$(OUT) : $(ROOT_DIR)/$(AOBJS_DIR)/$(OBJS) $(ROOT_DIR)/$(QOBJS_DIR)/$(OBJS) $(ROOT_DIR)/$(COBJS_DIR)/$(OBJS)
	$(CC) -o $@ $^	
$(ALIB) : $(ROOT_DIR)/$(AOBJS_DIR)/$(OBJS) $(ROOT_DIR)/$(QOBJS_DIR)/$(OBJS)
	ar rcs $@ $^
$(QLIB) : $(ROOT_DIR)/$(QOBJS_DIR)/$(OBJS)
	ar rcs $@ $^
	
	
	
