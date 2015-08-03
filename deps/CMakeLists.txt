file(GLOB_RECURSE src ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} pipeline/*.cpp)

set(PIPELINE_SRC ${src} CACHE INTERNAL "pipeline sources")
set(PIPELINE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "pipeline include dir")

add_subdirectory(catch)
add_subdirectory(deeplocalizer-tagger)
