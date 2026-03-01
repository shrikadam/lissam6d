cmake_minimum_required(VERSION 3.14)
project(SAM6D_EdgeTracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. OpenCV
find_package(OpenCV REQUIRED)

# 2. ONNX Runtime Paths (Based on our extraction step)
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 3. Project Includes
include_directories(include)

# 4. Build Executable
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/BaseInferencer.cpp
)

# 5. Link Libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    onnxruntime
)