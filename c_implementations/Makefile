# Makefile for building pandas and string C libraries

CC = clang
CFLAGS = -Wall -Wextra -O3 -fPIC -std=c11
INCLUDES = -Iinclude
LIB_DIR = lib
OBJ_DIR = obj

# Source files
PANDAS_SRCS = pandas/pandas_operations.c include/data_structures.c
STRING_SRCS = string/string_operations.c

# Object files
PANDAS_OBJS = $(PANDAS_SRCS:%.c=$(OBJ_DIR)/%.o)
STRING_OBJS = $(STRING_SRCS:%.c=$(OBJ_DIR)/%.o)

# Libraries
PANDAS_LIB = $(LIB_DIR)/libpandas_c.a
STRING_LIB = $(LIB_DIR)/libstring_c.a
PANDAS_SHARED = $(LIB_DIR)/libpandas_c.dylib
STRING_SHARED = $(LIB_DIR)/libstring_c.dylib

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SHARED_EXT = .so
    SHARED_FLAGS = -shared
endif
ifeq ($(UNAME_S),Darwin)
    SHARED_EXT = .dylib
    SHARED_FLAGS = -shared -install_name @rpath/libpandas_c.dylib
endif

.PHONY: all clean directories pandas string shared

all: directories pandas string shared

directories:
	@mkdir -p $(OBJ_DIR)/pandas $(OBJ_DIR)/string $(OBJ_DIR)/include $(LIB_DIR)

# Static libraries
pandas: $(PANDAS_LIB)
string: $(STRING_LIB)

# Shared libraries
shared: $(LIB_DIR)/libpandas_c$(SHARED_EXT) $(LIB_DIR)/libstring_c$(SHARED_EXT)

# Pandas static library
$(PANDAS_LIB): $(PANDAS_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# String static library
$(STRING_LIB): $(STRING_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# Pandas shared library
$(LIB_DIR)/libpandas_c$(SHARED_EXT): $(PANDAS_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) $(SHARED_FLAGS) -o $@ $^

# String shared library
$(LIB_DIR)/libstring_c$(SHARED_EXT): $(STRING_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) $(SHARED_FLAGS) -install_name @rpath/libstring_c$(SHARED_EXT) -o $@ $^

# Compile object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

install: all
	@echo "Libraries built in $(LIB_DIR)/"
	@echo "Headers in include/"
	@echo ""
	@echo "To use with FastPPI, set:"
	@echo "  export LIBRARY_PATH=\$$LIBRARY_PATH:$(PWD)/$(LIB_DIR)"
	@echo "  export C_INCLUDE_PATH=\$$C_INCLUDE_PATH:$(PWD)/include"

