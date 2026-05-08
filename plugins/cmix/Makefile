# plugins/cmix/Makefile — builds cmix.dll for 64-bit Windows via MinGW-w64.
#
# CMIX is heavy: peak RSS easily exceeds 25 GB on big inputs and encode
# time is hours/GB.  This codec is a niche "ratio over everything" choice.
#
# We omit CMIX's preprocessor (English-text dictionary, PHDA9) and its
# enwik9-specific reorderer; only the core context-mixing predictor +
# arithmetic coder are in the DLL.

CXX        := x86_64-w64-mingw32-g++
CXXFLAGS   := -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter \
              -ffunction-sections -fdata-sections \
              -DNDEBUG

SDK        := ../sdk
CXXFLAGS   += -I$(SDK) -I$(SDK)/CPP -include $(SDK)/StdAfx.h
LDFLAGS    := -shared -static-libgcc -static-libstdc++ \
              -Wl,--kill-at -Wl,--gc-sections
LIBS       := -loleaut32 -luuid

BUILD      := build
TARGET     := $(BUILD)/cmix.dll

CODECS_DIR ?= /mnt/l/Programs/7-Zip/Codecs

# CMIX core sources (everything except runner.cpp, the preprocessor, and
# the enwik9-preproc tool).
CMIX_DIR := upstream/src
CMIX_SRCS := \
    $(CMIX_DIR)/coder/decoder.cpp \
    $(CMIX_DIR)/coder/encoder.cpp \
    $(CMIX_DIR)/context-manager.cpp \
    $(CMIX_DIR)/contexts/bit-context.cpp \
    $(CMIX_DIR)/contexts/bracket-context.cpp \
    $(CMIX_DIR)/contexts/combined-context.cpp \
    $(CMIX_DIR)/contexts/context-hash.cpp \
    $(CMIX_DIR)/contexts/indirect-hash.cpp \
    $(CMIX_DIR)/contexts/interval-hash.cpp \
    $(CMIX_DIR)/contexts/interval.cpp \
    $(CMIX_DIR)/contexts/sparse.cpp \
    $(CMIX_DIR)/mixer/byte-mixer.cpp \
    $(CMIX_DIR)/mixer/lstm-layer.cpp \
    $(CMIX_DIR)/mixer/lstm.cpp \
    $(CMIX_DIR)/mixer/mixer-input.cpp \
    $(CMIX_DIR)/mixer/mixer.cpp \
    $(CMIX_DIR)/mixer/sigmoid.cpp \
    $(CMIX_DIR)/mixer/sse.cpp \
    $(CMIX_DIR)/models/bracket.cpp \
    $(CMIX_DIR)/models/byte-model.cpp \
    $(CMIX_DIR)/models/direct-hash.cpp \
    $(CMIX_DIR)/models/direct.cpp \
    $(CMIX_DIR)/models/indirect.cpp \
    $(CMIX_DIR)/models/fxcmv1.cpp \
    $(CMIX_DIR)/models/match.cpp \
    $(CMIX_DIR)/models/paq8.cpp \
    $(CMIX_DIR)/models/ppmd.cpp \
    $(CMIX_DIR)/predictor.cpp \
    $(CMIX_DIR)/states/nonstationary.cpp \
    $(CMIX_DIR)/states/run-map.cpp

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
    src/CmixCoder.cpp \
    src/Register.cpp \
    $(CMIX_SRCS)

OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(subst ../,__up__/,$(CXX_SRCS)))

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

define COMPILE_RULE
$(BUILD)/$(subst ../,__up__/,$(1:.cpp=.o)): $(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) -c $$< -o $$@
endef

$(foreach src,$(CXX_SRCS),$(eval $(call COMPILE_RULE,$(src))))

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	cp -v $(TARGET) "$(CODECS_DIR)/"
