# Example Makefile for RP2040 Pizza Detection with CMSIS-NN optimization

# Compiler and flags
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG

# Optional: Enable CMSIS-NN optimization (uncomment to enable)
CFLAGS += -DUSE_CMSIS_NN
# Define a flag to use our performance event logging wrapper
CFLAGS += -DPERFORMANCE_LOGGER_HAS_EVENTS

# RP2040 SDK path - update this to your SDK location
PICO_SDK_PATH = /path/to/pico-sdk

# CMSIS-NN path - update this to your CMSIS-NN location
CMSIS_PATH = /path/to/cmsis
CMSIS_NN_PATH = $(CMSIS_PATH)/CMSIS/NN

# Include paths
INCLUDES = -I$(PICO_SDK_PATH)/src/common/pico_stdlib/include \
           -I$(PICO_SDK_PATH)/src/rp2_common/hardware_gpio/include \
           -I$(PICO_SDK_PATH)/src/rp2_common/hardware_adc/include \
           -I$(CMSIS_PATH)/CMSIS/Core/Include \
           -I$(CMSIS_NN_PATH)/Include \
           -I.

# Source files
SRCS = main.c pizza_model.c pizza_preprocess.c pizza_temporal.c camera_utils.c
PERF_SRCS = performance_logger.c

# CMSIS-NN sources (only compiled if USE_CMSIS_NN is defined)
CMSIS_SRCS = pizza_model_cmsis.c pizza_benchmark.c

# Conditional compilation based on CMSIS-NN flag
ifneq (,$(findstring USE_CMSIS_NN,$(CFLAGS)))
    SRCS += $(CMSIS_SRCS)
    # Add CMSIS-NN core library files
    CMSIS_NN_SRCS = $(CMSIS_NN_PATH)/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c \
                    $(CMSIS_NN_PATH)/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast.c \
                    $(CMSIS_NN_PATH)/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c \
                    $(CMSIS_NN_PATH)/Source/FullyConnectedFunctions/arm_fully_connected_q7.c \
                    $(CMSIS_NN_PATH)/Source/ActivationFunctions/arm_relu_q7.c \
                    $(CMSIS_NN_PATH)/Source/SoftmaxFunctions/arm_softmax_q7.c \
                    $(CMSIS_NN_PATH)/Source/PoolingFunctions/arm_max_pool_s8.c \
                    $(CMSIS_NN_PATH)/Source/PoolingFunctions/arm_avgpool_s8.c
    SRCS += $(CMSIS_NN_SRCS)
endif

# Example application sources
EXAMPLE_SRCS = pizza_cmsis_example.c $(SRCS) $(PERF_SRCS)

# Test application sources
TEST_SRCS = pizza_cmsis_test.c $(SRCS) $(PERF_SRCS)

# Build targets
all: pizza_detector.uf2 pizza_cmsis_example.uf2 pizza_cmsis_test.uf2

pizza_detector.elf: main.c $(SRCS) $(PERF_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

pizza_detector.uf2: pizza_detector.elf
	elf2uf2 $< $@

pizza_cmsis_example.elf: $(EXAMPLE_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

pizza_cmsis_example.uf2: pizza_cmsis_example.elf
	elf2uf2 $< $@

pizza_cmsis_test.elf: $(TEST_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

pizza_cmsis_test.uf2: pizza_cmsis_test.elf
	elf2uf2 $< $@

benchmark: CFLAGS += -DBENCHMARK_MODE
benchmark: pizza_cmsis_test.uf2

clean:
	rm -f *.elf *.uf2
