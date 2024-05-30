cmake_minimum_required(VERSION 3.20)
project(MeasFakeRate VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# Find ROOT packages
find_package(ROOT REQUIRED COMPONENTS RooFit RooFitCore RooStats)

# glob files
file(GLOB SOURCES src/*.cc)
file(GLOB HEADERS include/*.h)
list(FILTER HEADERS EXCLUDE REGEX "LinkDef.h")
ROOT_GENERATE_DICTIONARY(G__MeasFakeRate 
    ${HEADERS} 
    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/include/LinkDef.h)

# add library
add_library(MeasFakeRate SHARED ${SOURCES} G__MeasFakeRate.cxx)
target_include_directories(MeasFakeRate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROOT_INCLUDE_DIRS})
target_link_libraries(MeasFakeRate PUBLIC ${ROOT_LIBRARIES})

# Install the library
install(TARGETS MeasFakeRate DESTINATION lib)

# Install .rootmap and .pcm files to the lib
set(ROOTMAP "${CMAKE_CURRENT_BINARY_DIR}/libMeasFakeRate.rootmap")
set(PCM "${CMAKE_CURRENT_BINARY_DIR}/libMeasFakeRate_rdict.pcm")
install(FILES ${ROOTMAP} ${PCM} DESTINATION lib)

