# plugins/zpaq/Makefile — builds zpaq.dll for 64-bit Windows via MinGW-w64.
#
# Usage:
#   make            # builds build/zpaq.dll
#   make install    # copies build/zpaq.dll to $(CODECS_DIR)
#   make clean
#
# Override the install path on the command line:
#   make install CODECS_DIR="/mnt/c/Program Files/7-Zip/Codecs"

CXX        := x86_64-w64-mingw32-g++
CXXFLAGS   := -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter \
              -ffunction-sections -fdata-sections
SDK        := ../sdk
CXXFLAGS   += -I$(SDK) -I$(SDK)/CPP -include $(SDK)/StdAfx.h
LDFLAGS    := -shared -static-libgcc -static-libstdc++ \
              -Wl,--kill-at -Wl,--gc-sections
LIBS       := -loleaut32 -luuid -lstdc++

BUILD      := build
TARGET     := $(BUILD)/zpaq.dll

CODECS_DIR ?= /mnt/l/Programs/7-Zip/Codecs

# Source files (paths relative to this Makefile)
SRCS := \
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
    src/ZpaqCoder.cpp \
    src/Register.cpp \
    upstream/libzpaq.cpp

# Map each source to a flattened .o path under $(BUILD)/, replacing
# ".." with "__up__" so we never write outside $(BUILD).
OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(subst ../,__up__/,$(SRCS)))

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# A separate compile rule per source — needed because the obj path no
# longer mirrors the source path (we substituted "../" → "__up__/").
define COMPILE_RULE
$(BUILD)/$(subst ../,__up__/,$(1:.cpp=.o)): $(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) -c $$< -o $$@
endef

$(foreach src,$(SRCS),$(eval $(call COMPILE_RULE,$(src))))

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	cp -v $(TARGET) "$(CODECS_DIR)/"
