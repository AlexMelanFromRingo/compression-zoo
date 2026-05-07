# plugins/zpaq/Makefile — builds zpaq.dll for 64-bit Windows via MinGW-w64.
#
# Usage:
#   make            # builds build/zpaq.dll
#   make install    # copies build/zpaq.dll to $(CODECS_DIR)
#   make clean
#
# Override the install path on the command line:
#   make install CODECS_DIR="/mnt/c/Program Files/7-Zip/Codecs"

CXX        ?= x86_64-w64-mingw32-g++
CXXFLAGS   ?= -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter
SDK        := ../sdk
CXXFLAGS   += -I$(SDK) -I$(SDK)/CPP -include $(SDK)/StdAfx.h
LDFLAGS    ?= -shared -static -static-libgcc -static-libstdc++ -Wl,--kill-at

BUILD      := build
TARGET     := $(BUILD)/zpaq.dll

CODECS_DIR ?= /mnt/l/Programs/7-Zip/Codecs

# Sources
SDK_SRCS := \
    $(SDK)/CPP/7zip/Compress/DllExportsCompress.cpp \
    $(SDK)/CPP/7zip/Compress/CodecExports.cpp \
    $(SDK)/CPP/7zip/Common/MethodId.cpp \
    $(SDK)/CPP/Common/MyVector.cpp \
    $(SDK)/CPP/Common/MyString.cpp \
    $(SDK)/CPP/Common/IntToString.cpp \
    $(SDK)/CPP/Common/NewHandler.cpp \
    $(SDK)/CPP/Common/StringConvert.cpp \
    $(SDK)/CPP/Common/UTFConvert.cpp \
    $(SDK)/CPP/Windows/PropVariant.cpp

PLUGIN_SRCS := \
    src/ZpaqCoder.cpp \
    src/Register.cpp

UPSTREAM_SRCS := upstream/libzpaq.cpp

SRCS := $(SDK_SRCS) $(PLUGIN_SRCS) $(UPSTREAM_SRCS)
OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS))

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	cp -v $(TARGET) "$(CODECS_DIR)/"
