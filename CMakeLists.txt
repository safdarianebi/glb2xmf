cmake_minimum_required(VERSION 3.10)
project(my_converter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/tinygltf
    ${PROJECT_SOURCE_DIR}/external/pugixml
    ${HDF5_INCLUDE_DIRS}  # Use found HDF5 includes
)

# Add source files
set(SOURCES
    src/main.cpp
    external/pugixml/pugixml.cpp
    external/tinygltf/tiny_gltf.cc
)

# Create executable
add_executable(glb2xmf ${SOURCES})

# Set runtime output directory
set_target_properties(glb2xmf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Set include directories for the target
target_include_directories(glb2xmf PRIVATE
    ${PROJECT_SOURCE_DIR}/external/tinygltf
    ${PROJECT_SOURCE_DIR}/external/pugixml
    ${HDF5_INCLUDE_DIRS}
)

# Link HDF5 libraries
target_link_libraries(glb2xmf PRIVATE
    ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES}
)

# Ensure output directory exists
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/output)
