# Toolchain
CXX=arm-none-eabi-g++
CC=arm-none-eabi-gcc
AR=arm-none-eabi-ar

# Compiler and Linker Flags
CFLAGS = -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Os -ffunction-sections -fdata-sections
CXXFLAGS = $(CFLAGS) -std=c++11
LDFLAGS = -Tstm32_flash.ld -Wl,--gc-sections -Wl,-Map=output.map

# Include paths for TFLite Micro and CMSIS
INCLUDES = -I./tensorflow/lite/micro -I./CMSIS/Core/Include -I./CMSIS/NN/Include

# Source files for TFLite Micro and the project
SRCS = main.cpp tensorflow/lite/micro/micro_interpreter.cc \
       tensorflow/lite/micro/kernels/all_ops_resolver.cc \
       tensorflow/lite/micro/kernels/*.cc

# Output file
OUTFILE = tflite_micro_project.elf

# Build Rules
all: $(OUTFILE)

$(OUTFILE): $(SRCS)
    $(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(OUTFILE) $(LDFLAGS)

clean:
    del /Q $(OUTFILE) *.o
