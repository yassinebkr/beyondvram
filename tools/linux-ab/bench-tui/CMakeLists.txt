cmake_minimum_required(VERSION 3.16)
project(bench-tui CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

set(BENCH_TUI_SOURCES
    src/config.cpp
    src/collectors.cpp
    src/format.cpp
    src/fsutil.cpp
    src/json.cpp
    src/run_manager.cpp)

add_executable(bench-tui src/main.cpp src/ui_tabs.cpp ${BENCH_TUI_SOURCES})
target_include_directories(bench-tui PRIVATE third_party src)
target_link_libraries(bench-tui PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

add_executable(bench-tui-tests tests/tests.cpp ${BENCH_TUI_SOURCES})
target_include_directories(bench-tui-tests PRIVATE third_party src)
target_link_libraries(bench-tui-tests PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

enable_testing()
add_test(NAME unit COMMAND bench-tui-tests)
