file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
list(REMOVE_ITEM src "tagger.cpp" "preprocess.cpp" )
file(GLOB hdr ${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/*.h)
file(GLOB_RECURSE ui RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ui)
file(GLOB_RECURSE qrc RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.qrc)

include_directories(${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/)

qt5_wrap_cpp(MOC_HDR ${hdr})

qt5_add_resources(UI_RESOURCES ${qrc})
qt5_wrap_ui(UI_HEADERS ${ui})

add_library(deeplocalizer-tagger
    ${src} ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(deeplocalizer-tagger ${libs})

add_executable(tagger "tagger.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

target_link_libraries(tagger deeplocalizer-tagger)

add_executable(bb_preprocess "preprocess.cpp" ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(bb_preprocess deeplocalizer-tagger)

install (TARGETS bb_preprocess  deeplocalizer-tagger
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
