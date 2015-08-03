file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
list(REMOVE_ITEM src "tagger.cpp" "add_border.cpp" "generate_proposals.cpp"
                     "generate_dataset.cpp" "caffe_stats.cpp")
file(GLOB hdr ${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/*.h)
file(GLOB_RECURSE ui RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ui)
file(GLOB_RECURSE qrc RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.qrc)

include_directories(${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/)

set(src ${src} ${PIPELINE_SRC})

qt5_wrap_cpp(MOC_HDR ${hdr})


qt5_add_resources(UI_RESOURCES ${qrc})
qt5_wrap_ui(UI_HEADERS ${ui})

add_library(deeplocalizer-tagger
    ${src} ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

add_executable(tagger "tagger.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

target_link_libraries(tagger ${tagger-libs})

add_executable(add_border "add_border.cpp" ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(add_border ${tagger-libs})

add_executable(generate_proposals "generate_proposals.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

target_link_libraries(generate_proposals ${tagger-libs})


add_executable(generate_dataset "generate_dataset.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})
target_link_libraries(generate_dataset ${tagger-libs})

