
enable_testing(true)
add_subdirectory(testdata)
add_subdirectory(scripts)


include_directories(
    ${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/
)

file(GLOB tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} Test*.cpp)

foreach(test ${tests})
    get_filename_component(name ${test} NAME_WE)
    add_executable(${name} ${test})
    target_link_libraries(${name} ${test-libs})
    set(test_bin ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(${name} ${test_bin})
endforeach()
