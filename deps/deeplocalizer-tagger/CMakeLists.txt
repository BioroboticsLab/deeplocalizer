cmake_minimum_required(VERSION 2.8.8)
project(deeplocalizer_classifier CXX)
include(ExternalProject)
find_package(Git REQUIRED)
ExternalProject_Add(
    deeplocalizer_classifier
    PREFIX ${CMAKE_BINARY_DIR}/deeplocalizer-classifier
    GIT_REPOSITORY https://github.com/BioroboticsLab/deeplocalizer-classifier.git
    TIMEOUT 10
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Get_Property(deeplocalizer_classifier BINARY_DIR)
ExternalProject_Get_Property(deeplocalizer_classifier SOURCE_DIR)
set(DEEP_CLASSIFIER_INCLUDE_DIR ${SOURCE_DIR}/include/deeplocalizer/classifier CACHE INTERNAL "Path to include folder for Catch")
set(DEEP_CLASSIFIER_LD_DIR ${BINARY_DIR}/source/classifier CACHE INTERNAL "Path to include folder for Catch")

