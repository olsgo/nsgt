cmake_minimum_required(VERSION 3.15)

project(spectrogram_texture_plugin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(flutter REQUIRED)

# Plugin sources
set(PLUGIN_SOURCES
    "spectrogram_texture_plugin.cpp"
    "spectrogram_texture_plugin.h"
)

# Create plugin library
add_library(${PROJECT_NAME} SHARED ${PLUGIN_SOURCES})

# Link Flutter embedder
target_link_libraries(${PROJECT_NAME} PRIVATE flutter flutter_wrapper_app)

# Link DirectX libraries for Windows GPU integration
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        d3d11.lib
        dxgi.lib
        dxguid.lib
    )
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    FLUTTER_PLUGIN_IMPL
    UNICODE
    _UNICODE
)

# Set plugin properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    LINK_WHAT_YOU_USE TRUE
)

# Install plugin
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)