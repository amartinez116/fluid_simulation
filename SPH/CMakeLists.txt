cmake_minimum_required(VERSION 2.8.12.2)
project(SPH)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

set(SOURCE_FILES main.cpp)
add_executable(SPH main.cpp fluid.cpp particle.cpp utility.h)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

if (APPLE)
    target_link_libraries(SPH "-framework OpenGL")
    target_link_libraries(SPH "-framework GLUT")
endif()