cmake_minimum_required(VERSION 3.10)
project(Prun LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(prun
    main.cpp
    layers/Linear.cpp
    layers/FeedForward.cpp
    model/TransformerBlock.cpp
    model/MoE.cpp
)