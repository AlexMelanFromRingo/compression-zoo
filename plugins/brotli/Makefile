# plugins/brotli/Makefile — builds brotli.dll for 64-bit Windows via MinGW-w64.
#
# Usage:
#   make            # builds build/brotli.dll
#   make install    # copies build/brotli.dll to $(CODECS_DIR)
#   make clean

CXX        := x86_64-w64-mingw32-g++
CC         := x86_64-w64-mingw32-gcc
COMMON_FLAGS := -O2 -Wall -Wno-unused-parameter \
                -ffunction-sections -fdata-sections
CXXFLAGS   := $(COMMON_FLAGS) -std=c++14
CFLAGS     := $(COMMON_FLAGS) -std=c99

SDK        := ../sdk
BROTLI_INC := -Iupstream/c/include
CXXFLAGS   += -I$(SDK) -I$(SDK)/CPP $(BROTLI_INC) -include $(SDK)/StdAfx.h
CFLAGS     += $(BROTLI_INC)
LDFLAGS    := -shared -static-libgcc -static-libstdc++ \
              -Wl,--kill-at -Wl,--gc-sections
LIBS       := -loleaut32 -luuid

BUILD      := build
TARGET     := $(BUILD)/brotli.dll

CODECS_DIR ?= /mnt/l/Programs/7-Zip/Codecs

# C++ sources: SDK + our coder.
CXX_SRCS := \
    $(SDK)/CPP/7zip/Compress/DllExportsCompress.cpp \
    $(SDK)/CPP/7zip/Compress/CodecExports.cpp \
    $(SDK)/CPP/7zip/Common/MethodId.cpp \
    $(SDK)/CPP/Common/MyVector.cpp \
    $(SDK)/CPP/Common/MyString.cpp \
    $(SDK)/CPP/Common/IntToString.cpp \
    $(SDK)/CPP/Common/NewHandler.cpp \
    $(SDK)/CPP/Common/StringConvert.cpp \
    $(SDK)/CPP/Common/UTFConvert.cpp \
    $(SDK)/CPP/Windows/PropVariant.cpp \
    src/BrotliCoder.cpp \
    src/Register.cpp

# C sources: all of upstream/c/{common,dec,enc}/*.c.
C_SRCS := $(sort $(wildcard upstream/c/common/*.c) \
                 $(wildcard upstream/c/dec/*.c) \
                 $(wildcard upstream/c/enc/*.c))

OBJS := \
    $(patsubst %.cpp,$(BUILD)/%.o,$(subst ../,__up__/,$(CXX_SRCS))) \
    $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

define COMPILE_CXX
$(BUILD)/$(subst ../,__up__/,$(1:.cpp=.o)): $(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) -c $$< -o $$@
endef

define COMPILE_C
$(BUILD)/$(1:.c=.o): $(1)
	@mkdir -p $$(dir $$@)
	$(CC) $(CFLAGS) -c $$< -o $$@
endef

$(foreach src,$(CXX_SRCS),$(eval $(call COMPILE_CXX,$(src))))
$(foreach src,$(C_SRCS),$(eval $(call COMPILE_C,$(src))))

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	cp -v $(TARGET) "$(CODECS_DIR)/"
