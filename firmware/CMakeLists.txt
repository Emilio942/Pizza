
# CMakeLists.txt for Pizza Verifier RP2040 Integration
cmake_minimum_required(VERSION 3.12)

# Include CMSIS-NN
set(CMSIS_NN_PATH "${CMAKE_CURRENT_LIST_DIR}/../CMSIS-NN")
include_directories(${CMSIS_NN_PATH}/Include)

# Pizza Verifier sources
set(PIZZA_VERIFIER_SOURCES
    pizza_verifier_inference.c
    pizza_verifier_example.c
)

# CMSIS-NN library sources
file(GLOB CMSIS_NN_SOURCES 
    "${CMSIS_NN_PATH}/Source/FullyConnectedFunctions/*.c"
    "${CMSIS_NN_PATH}/Source/ActivationFunctions/*.c"
    "${CMSIS_NN_PATH}/Source/BasicMathFunctions/*.c"
)

# Create pizza verifier library
add_library(pizza_verifier STATIC
    ${PIZZA_VERIFIER_SOURCES}
    ${CMSIS_NN_SOURCES}
)

target_include_directories(pizza_verifier PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMSIS_NN_PATH}/Include
)

# Compiler optimizations for RP2040
target_compile_options(pizza_verifier PRIVATE
    -O3
    -mthumb
    -mcpu=cortex-m0plus
    -flto
)

# Example executable
add_executable(pizza_verifier_example
    pizza_verifier_example.c
)

target_link_libraries(pizza_verifier_example pizza_verifier)
