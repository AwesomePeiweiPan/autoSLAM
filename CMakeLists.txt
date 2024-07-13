cmake_minimum_required(VERSION 3.5)
project(slam_in_auto_driving)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)                                      #C++的标准为C++17
set(CMAKE_CXX_FLAGS "-w")                                       #编译器抑制所有的警告

#-02:启用编译器的第二级优化。级别越高编译，性能越好，编译的时间越长
#-g :告诉编译器生成调试信息，即使在release模式下，也可以在调试器中查看源代码
#-ggbd :-g的扩展版本，编译器生成更多的信息，特别是对gdb优化的调试信息
# ${CMAKE_CXX_FLAGS}: 包含其他通用的编译选项，如果其他地方设置了全局的编译选项，也会被包含在这里
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -ggdb ${CMAKE_CXX_FLAGS}")  
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")

#禁用catkin测试功能: 这个功能是使用Catkin构建系统来进行自动化测试，确保ROS包和节点的正确性
add_definitions("-DCATKIN_ENABLE_TESTING=0")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin) #可执行文件的输出放在当前的bin
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib) #动态库文件输出设置 为项目根目录下的 lib 目录 
                                                              #.so文件，动态链接，关心路径，小心依赖冲突
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib) #静态库文件输出设置 为项目根目录下的 lib 目录
                                                              #.lib, .a文件，已经被预编译，每个都有副本，不关心路径

#看系统了
option(BUILD_WITH_UBUNTU1804 OFF)


include(cmake/packages.cmake) #包含一个 packages.cmake 的CMake 文件，包含了第三方库的 查找和配置
include_directories(${PROJECT_SOURCE_DIR}/src) # 添加 项目根目录下的 src 作为 头文件的搜索路径中
include_directories(${PROJECT_SOURCE_DIR})     # 添加 根目录 作为 头文件的搜索路径中
include_directories(${PROJECT_SOURCE_DIR}/build/devel/include)  # 添加 此目录 作为 头文件的搜索目录中

add_subdirectory(src) #添加 src 目录到子目录构建系统中，CMake将处理 src 下的 CMakeList文件，并构建项目
