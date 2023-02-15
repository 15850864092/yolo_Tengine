SET(CMAKE_SYSTEM_NAME Linux)
#set(CMAKE_SYSTEM_PROCESSOR arm)
SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")

#SET(TENGINE_TOOLCHAIN_FLAG -march=armv8-a -mfloat-abi=hard -mfpu=neon-vfpv4)
set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -rdynamic -O3 -fPIC -ggdb -std=c++11")


#1.cmake verson，指定cmake版本
cmake_minimum_required(VERSION 3.2)
#2.project name，指定项目的名称，一般和项目的文件夹名称对应
PROJECT(yolo_Tengine)

add_compile_options(-std=c++11)

#3.head file path，头文件目录
INCLUDE_DIRECTORIES(/home/Park/Tengine/build/install/include/)
INCLUDE_DIRECTORIES(/home/opencv_4.5.5/include/opencv4/)
INCLUDE_DIRECTORIES(/home/Park/yolo_Tengine/)


#4.source directory，源文件目录
#AUX_SOURCE_DIRECTORY(src DIR_SRCS)
#AUX_SOURCE_DIRECTORY(/home/Park/yolo_Tengine/ DIR_SRCS)

#5.set environment variable，设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
#SET(TEST_PATH ${DIR_SRCS})

LINK_DIRECTORIES("/home/Park/Tengine/build/install/lib")
LINK_DIRECTORIES("/home/opencv_4.5.5/lib/")
#6.add executable file，添加要编译的可执行文件
#ADD_EXECUTABLE(${PROJECT_NAME} ${TEST_PATH})
add_executable(yolo_Tengine main.cpp yolov5m6.cpp)

#7.add link library，添加可执行文件所需要的库，比如我们用到了libm.so（命名规则：lib+name+.so），就添加该库的名称
TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_core.so)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_imgproc.so)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_imgcodecs.so)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} libtengine-lite.so)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} -lpthread -ldl)