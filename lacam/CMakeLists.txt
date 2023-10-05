cmake_minimum_required(VERSION 3.16)
file(GLOB SRCS "./src/*.cpp")
project(lacam)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} STATIC ${SRCS})
# target_compile_options(${PROJECT_NAME} PUBLIC -O3 -Wall -mtune=native) # Removing opimizations for now.
target_compile_options(${PROJECT_NAME} PUBLIC)  # Version without optimizations.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE ./include)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
