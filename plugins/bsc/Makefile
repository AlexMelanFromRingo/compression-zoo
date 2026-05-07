# plugins/bsc/Makefile — builds bsc.dll for 64-bit Windows via MinGW-w64.
#
# Usage:
#   make            # builds build/bsc.dll
#   make install    # copies build/bsc.dll to $(CODECS_DIR)
#   make clean

CXX        := x86_64-w64-mingw32-g++
CC         := x86_64-w64-mingw32-gcc
COMMON_FLAGS := -O2 -Wall -Wextra -Wno-unused-parameter \
                -ffunction-sections -fdata-sections \
                -DLIBBSC_SORT_TRANSFORM_SUPPORT \
                -DLIBBSC_ALLOW_UNALIGNED_ACCESS \
                -DLIBBSC_PLATFORM_HEADERS=1
CXXFLAGS   := $(COMMON_FLAGS) -std=c++14
CFLAGS     := $(COMMON_FLAGS) -std=c11

SDK        := ../sdk
CXXFLAGS   += -I$(SDK) -I$(SDK)/CPP -include $(SDK)/StdAfx.h
LDFLAGS    := -shared -static-libgcc -static-libstdc++ \
              -Wl,--kill-at -Wl,--gc-sections
LIBS       := -loleaut32 -luuid -lstdc++

BUILD      := build
TARGET     := $(BUILD)/bsc.dll

CODECS_DIR ?= /mnt/l/Programs/7-Zip/Codecs

# C++ sources (SDK + libbsc + our wrapper).
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
    upstream/libbsc/adler32/adler32.cpp \
    upstream/libbsc/bwt/bwt.cpp \
    upstream/libbsc/coder/coder.cpp \
    upstream/libbsc/coder/qlfc/qlfc.cpp \
    upstream/libbsc/coder/qlfc/qlfc_model.cpp \
    upstream/libbsc/filters/detectors.cpp \
    upstream/libbsc/filters/preprocessing.cpp \
    upstream/libbsc/libbsc/libbsc.cpp \
    upstream/libbsc/lzp/lzp.cpp \
    upstream/libbsc/platform/platform.cpp \
    upstream/libbsc/st/st.cpp \
    src/BscCoder.cpp \
    src/Register.cpp

C_SRCS := \
    upstream/libbsc/bwt/libsais/libsais.c

OBJS := \
    $(patsubst %.cpp,$(BUILD)/%.o,$(subst ../,__up__/,$(CXX_SRCS))) \
    $(patsubst %.c,$(BUILD)/%.o,$(subst ../,__up__/,$(C_SRCS)))

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
$(BUILD)/$(subst ../,__up__/,$(1:.c=.o)): $(1)
	@mkdir -p $$(dir $$@)
	$(CC) $(CFLAGS) -c $$< -o $$@
endef

$(foreach src,$(CXX_SRCS),$(eval $(call COMPILE_CXX,$(src))))
$(foreach src,$(C_SRCS),$(eval $(call COMPILE_C,$(src))))

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	cp -v $(TARGET) "$(CODECS_DIR)/"
