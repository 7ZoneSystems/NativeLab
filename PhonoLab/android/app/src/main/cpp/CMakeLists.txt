cmake_minimum_required(VERSION 3.22.1)
project("runner")

add_library(runner SHARED runner.cpp)

target_link_libraries(runner
    log
    android)
