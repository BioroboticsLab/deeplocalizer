file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
list(REMOVE_ITEM src "tagger.cpp" "preprocess.cpp" "generate_proposals.cpp"
                     "generate_dataset.cpp" "caffe_stats.cpp")
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

add_executable(preprocess "preprocess.cpp" ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(preprocess deeplocalizer-tagger)

add_executable(generate_proposals "generate_proposals.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

target_link_libraries(generate_proposals deeplocalizer-tagger)


add_executable(generate_dataset "generate_dataset.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(generate_dataset deeplocalizer-tagger)

