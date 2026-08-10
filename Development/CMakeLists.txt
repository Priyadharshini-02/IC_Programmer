cmake_minimum_required(VERSION 3.10)
project(PCBCameraGrabber)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenCV ────────────────────────────────────────────────────────────────
set(OpenCV_DIR "C:/opencv_custom/build")
find_package(OpenCV REQUIRED)

# ── Pylon ─────────────────────────────────────────────────────────────────
set(PYLON_ROOT "C:/Program Files/Basler/pylon")

include_directories(
    ${PYLON_ROOT}/Development/include
)

link_directories(
    ${PYLON_ROOT}/Development/lib/x64
)

# ── camera_grabber_lib.dll ────────────────────────────────────────────────
add_library(camera_grabber_lib SHARED camera_grabber_lib.cpp)

target_compile_definitions(camera_grabber_lib PRIVATE BUILDING_DLL)

target_include_directories(camera_grabber_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_grabber_lib PRIVATE
    ${OpenCV_LIBS}
    PylonBase_v11
    PylonUtility_v11
    GenApi_MD_VC142_v3_5_Basler_pylon_v1
)

# ── Executable ────────────────────────────────────────────────────────────
add_executable(camera_grabber camera_grabber.cpp)

target_include_directories(camera_grabber PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_grabber PRIVATE
    camera_grabber_lib
    ${OpenCV_LIBS}
)

message(STATUS "OpenCV : ${OpenCV_VERSION}")
