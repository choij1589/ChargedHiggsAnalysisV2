cmake_minimum_required(VERSION 3.20)
project(SignalRegionStudyV1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROOT
find_package(ROOT REQUIRED COMPONENTS RIO Tree)

# Include ROOT headers
include_directories(${ROOT_INCLUDE_DIRS})

# Generate ROOT dictionary
file(GLOB HEADERS include/*.h)
list(FILTER HEADERS EXCLUDE REGEX "LinkDef.h")
ROOT_GENERATE_DICTIONARY(G__SignalRegionStudyV1
    ${HEADERS} 
    LINKDEF include/LinkDef.h 
)

# Define the library with source files
file(GLOB SOURCES src/*.cc)
add_library(SignalRegionStudyV1 SHARED 
    ${SOURCES} 
    G__SignalRegionStudyV1.cxx
)

target_include_directories(SignalRegionStudyV1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(SignalRegionStudyV1 PUBLIC ${ROOT_LIBRARIES})

install(TARGETS SignalRegionStudyV1 DESTINATION lib)

# Install .rootmap and .pcm files to the lib
set(ROOTMAP "${CMAKE_CURRENT_BINARY_DIR}/libSignalRegionStudyV1.rootmap")
set(PCM "${CMAKE_CURRENT_BINARY_DIR}/libSignalRegionStudyV1_rdict.pcm")
install(FILES ${ROOTMAP} ${PCM} DESTINATION lib)
