cmake_minimum_required(VERSION 2.8.12.2)
project(SPH)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenMP REQUIRED CXX)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

set(SOURCE_FILES main.cpp)
cuda_add_executable(SPH main.cpp fluid.cpp fluid_cuda.cu particle.cpp utility.h)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

if (APPLE)
    target_link_libraries(SPH "-framework OpenGL")
    target_link_libraries(SPH "-framework GLUT")
endif()
