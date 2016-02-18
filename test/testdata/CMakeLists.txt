
file(GLOB all_images RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.png *.PNG *.jpg *.JPEG *.jpeg *.txt  *.json)

foreach(image ${all_images})
    configure_file(${image} ${CMAKE_CURRENT_BINARY_DIR}/${image} COPYONLY)
endforeach()
