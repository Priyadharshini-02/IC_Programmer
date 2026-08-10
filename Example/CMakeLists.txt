cmake_minimum_required(VERSION 3.10)
project(PCBCameraGrabber)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenCV ────────────────────────────────────────────────────────────────
set(OpenCV_DIR "C:/opencv_custom/build")
find_package(OpenCV REQUIRED)

# ── Paths to the pre-built DLL (adjust if yours live elsewhere) ───────────
set(CG_DLL_DIR  "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Folder containing camera_grabber_lib.dll")
set(CG_IMP_LIB  "${CG_DLL_DIR}/camera_grabber_lib.lib"  CACHE FILEPATH "Import lib (.lib)")
set(CG_DLL      "${CG_DLL_DIR}/camera_grabber_lib.dll"  CACHE FILEPATH "Runtime DLL (.dll)")

# ── Imported target that wraps the pre-built DLL ──────────────────────────
add_library(camera_grabber_lib SHARED IMPORTED)

set_target_properties(camera_grabber_lib PROPERTIES
    IMPORTED_IMPLIB   "${CG_IMP_LIB}"   # .lib  — used at link time
    IMPORTED_LOCATION "${CG_DLL}"       # .dll  — used for the POST_BUILD copy
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── example executable ────────────────────────────────────────────────────
add_executable(example example.cpp)

target_include_directories(example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(example PRIVATE
    camera_grabber_lib
    ${OpenCV_LIBS}
)

# Copy DLL next to the exe so it is found at runtime
add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CG_DLL}"
        $<TARGET_FILE_DIR:example>
    COMMENT "Copying camera_grabber_lib.dll -> output dir"
)

message(STATUS "OpenCV          : ${OpenCV_VERSION}")
message(STATUS "CG import lib   : ${CG_IMP_LIB}")
message(STATUS "CG runtime DLL  : ${CG_DLL}")