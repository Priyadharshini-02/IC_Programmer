/**
 * camera_grabber_api.h  —  Public API for camera_grabber_lib.dll  (v3)
 *
 * Changes from v3
 * ---------------
 *   • CG_SetROI added — store the IC bounding-box once; used by both
 *     CG_SetReference and CG_InspectIC automatically.
 *   • CG_SetReference now runs a YOLO presence check (Python --mode
 *     check_roi) before accepting the frame; returns CG_ERR_NO_BOARD on
 *     rejection.  Saves the ROI crop as ref\current_reference.png and
 *     writes ref\roi_coords.txt  (all previously done in camera_grabber.cpp).
 *   • CG_SetReferenceFromFrame added — same as CG_SetReference but takes
 *     a caller-supplied frame (for use in callbacks).
 *   • CG_InspectIC / CG_InspectICFromFrame are now FULL inspections: they
 *     save the test crop, launch Python integrated_detector.py --mode
 *     interactive, parse the output, and fill CG_DetectionResult.  The
 *     entire pipeline that was previously in camera_grabber.cpp
 *     (checkICPresenceInROI, setReference, runFullInspection) is now
 *     inside the DLL.
 *   • CG_GetLastICResult added — retrieve the rich CG_ICResult (pin
 *     counts, rotation, etc.) after a CG_InspectIC call.
 *   • camera_grabber.cpp is no longer required to distribute to the
 *     software team.  Ship: DLL + camera_grabber_api.h + Python files.
 *
 * Use-cases
 * ---------
 *   1. LIVE FRAMES       — CG_StartStream / CG_SetFrameCallback
 *   2. SAVE FRAME        — CG_SaveFrame / CG_SaveFrameToPath
 *   3. BOARD PRESENCE    — CG_GetBoardPresence (OpenCV-only, no Python)
 *   4. SET ROI           — CG_SetROI (call once after user draws box)
 *   5. SET REFERENCE     — CG_SetReference / CG_SetReferenceFromFrame
 *                          (YOLO pre-check + saves crop + roi_coords.txt)
 *   6. FULL INSPECTION   — CG_InspectIC / CG_InspectICFromFrame
 *                          (crops frame → Python → CG_DetectionResult)
 *                          Rich result via CG_GetLastICResult()
 *
 * Lifecycle
 * ---------
 *   CG_Initialize()
 *       └─► CG_StartStream()
 *               ├─► CG_SetFrameCallback()
 *               ├─► CG_SaveFrame()
 *               ├─► CG_GetBoardPresence()
 *               ├─► CG_SetROI(x, y, w, h)        // draw ROI first
 *               ├─► CG_SetReference()             // YOLO check + save ref
 *               ├─► CG_InspectIC(...)             // full Python pipeline
 *               ├─► CG_GetLastICResult()          // rich details
 *               └─► CG_StopStream()
 *   CG_Shutdown()
 */

#pragma once

#ifdef _WIN32
  #ifdef BUILDING_DLL
    #define CG_API __declspec(dllexport)
  #else
    #define CG_API __declspec(dllimport)
  #endif
#else
  #define CG_API __attribute__((visibility("default")))
#endif

#ifndef _UINT8_T_DEFINED
  typedef unsigned char  uint8_t;
  #define _UINT8_T_DEFINED
#endif
#ifndef _UINT32_T_DEFINED
  typedef unsigned int   uint32_t;
  #define _UINT32_T_DEFINED
#endif
#ifndef _INT32_T_DEFINED
  typedef int            int32_t;
  #define _INT32_T_DEFINED
#endif
#ifndef _INT64_T_DEFINED
  #ifdef _MSC_VER
    typedef __int64      int64_t;
  #else
    typedef long long    int64_t;
  #endif
  #define _INT64_T_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
//  Core frame descriptor
// ============================================================================

typedef struct CG_Frame
{
    uint8_t* data;          ///< BGR pixel buffer (DLL-owned in callback, caller-owned in GetFrame)
    int32_t  width;         ///< Width  in pixels
    int32_t  height;        ///< Height in pixels
    int32_t  step;          ///< Row stride in bytes (>= width * 3)
    int64_t  timestamp_ms;  ///< Wall-clock ms since Unix epoch (UTC)
    uint32_t frame_index;   ///< Monotonically increasing grab counter
} CG_Frame;

// ============================================================================
//  Callback type
// ============================================================================

/**
 * CG_FrameCallback
 * Invoked on the internal capture thread for every grabbed frame.
 * @param frame  Valid only during the callback.
 * @param user   Opaque pointer supplied to CG_SetFrameCallback.
 *
 * *** Do NOT call CG_StopStream / CG_Shutdown from inside the callback. ***
 */
typedef void (*CG_FrameCallback)(const CG_Frame* frame, void* user);

// ============================================================================
//  Board-level presence
// ============================================================================

typedef struct CG_BoardPresence
{
    int    IC_present;  ///< Non-zero if a board/component was detected
    double edge_ratio;  ///< Edge pixel fraction (0.0 – 1.0)
    double stddev;      ///< Grayscale standard deviation
} CG_BoardPresence;

// ============================================================================
//  Lightweight IC inspection result  (DLL-side, no Python)
// ============================================================================

#define CG_MAX_COMPONENTS 512

/**
 * CG_Component
 * Per-IC result filled by the DLL-side CG_InspectIC / CG_InspectICFromFrame.
 *
 *   cls    — always "IC"
 *   status — one of:
 *              "present"           all checks passed
 *              "absent"            IC not detected in ROI
 *              "wrong_orientation" polarity dot in wrong quadrant
 *              "pin_damaged"       bent / missing / bridged pin detected
 */
typedef struct CG_Component
{
    int32_t id;
    float   x1, y1, x2, y2;    ///< ROI in full-frame pixels
    float   score;              ///< 1.0 = present, 0.0 = absent
    float   similarity;         ///< Reserved (0.0)
    char    cls[32];            ///< "IC"
    char    status[32];         ///< See above
} CG_Component;

/**
 * CG_DetectionResult
 * Returned by CG_InspectIC / CG_InspectICFromFrame (DLL lightweight stub).
 * For the full Python-based result use CG_ICResult (see below).
 */
typedef struct CG_DetectionResult
{
    int          present;
    int          pindamage;
    int          wrong_orientation;
    CG_Component components[CG_MAX_COMPONENTS];
} CG_DetectionResult;

// ============================================================================
//  Full inspection result  (Python-based, produced by camera_grabber.cpp)
// ============================================================================

/**
 * CG_ICStatus
 * High-level verdict returned in CG_ICResult.status after a full Python
 * inspection orchestrated by runFullInspection() in camera_grabber.cpp.
 */
typedef enum CG_ICStatus
{
    CG_IC_PASS              = 0,  ///< All checks passed
    CG_IC_ABSENT            = 1,  ///< IC not detected by YOLO
    CG_IC_WRONG_ORIENTATION = 2,  ///< IC rotated beyond tolerance
    CG_IC_PIN_DAMAGED       = 3,  ///< Bent / missing / bridged pin
    CG_IC_EXTRA             = 4,  ///< Unexpected IC in frame
    CG_IC_ERROR             = 5,  ///< Python / I/O failure
} CG_ICStatus;

/**
 * CG_ICResult
 * Rich result struct filled by runFullInspection() in camera_grabber.cpp.
 * Carries per-fault counts and rotation details parsed from Python output.
 */
typedef struct CG_ICResult
{
    CG_ICStatus status;

    // Presence
    int   present_count;            ///< ICs matched against reference
    int   absent_count;             ///< ICs expected but missing
    int   extra_count;              ///< Unexpected ICs detected

    // Orientation
    int   wrong_orientation_count;  ///< ICs with wrong polarity dot quadrant
    float rotation_deg;             ///< Measured rotation in degrees
    int   rotation_step;            ///< Nearest 90° step

    // Pin health
    int   pin_fail_count;           ///< Total pin-fault events
    int   pin_missing;              ///< Missing-pin count
    int   pin_bent;                 ///< Bent-pin count
    int   pin_bridged;              ///< Bridged-pin count

    char  message[256];             ///< Human-readable summary
} CG_ICResult;

// ============================================================================
//  Status / error codes
// ============================================================================

typedef enum CG_Status
{
    CG_OK                  =  0,
    CG_ERR_ALREADY_INIT    = -1,    ///< CG_Initialize called twice
    CG_ERR_NOT_INIT        = -2,    ///< Called before CG_Initialize
    CG_ERR_NO_CAMERA       = -3,    ///< Basler camera not found / Pylon not built
    CG_ERR_STREAM_ACTIVE   = -4,    ///< Operation invalid while streaming
    CG_ERR_STREAM_INACTIVE = -5,    ///< Stream not started
    CG_ERR_GRAB_FAILED     = -6,    ///< Frame acquisition error
    CG_ERR_SAVE_FAILED     = -7,    ///< imwrite / CopyFile failed
    CG_ERR_NO_REFERENCE    = -8,    ///< Reference not set
    CG_ERR_NO_BOARD        = -10,   ///< No board in frame
    CG_ERR_INVALID_ARG     = -11,   ///< NULL pointer or bad parameter
    CG_ERR_PYTHON_FAILED   = -12,   ///< Python subprocess returned non-zero
    CG_ERR_PARSE_FAILED    = -13,   ///< Could not parse Python output file
} CG_Status;

// ============================================================================
//  Lifecycle
// ============================================================================

/**
 * CG_Initialize
 * Must be the first CG_ call.
 * @return CG_OK or CG_ERR_ALREADY_INIT.
 */
CG_API CG_Status CG_Initialize(void);

/**
 * CG_Shutdown
 * Stops the stream and releases all resources.
 * Call once when your application exits.
 */
CG_API void CG_Shutdown(void);

// ============================================================================
//  Streaming
// ============================================================================

/** Open the first Basler camera and start the grab thread. */
CG_API CG_Status CG_StartStream(void);

/** Stop acquisition and close the camera. Safe to call if not streaming. */
CG_API CG_Status CG_StopStream(void);

/** @return Non-zero if the grab thread is running. */
CG_API int CG_IsStreaming(void);

// ============================================================================
//  Callback
// ============================================================================

/**
 * CG_SetFrameCallback
 * Register a function called on every new frame. Pass NULL to unregister.
 */
CG_API CG_Status CG_SetFrameCallback(CG_FrameCallback cb, void* user);

// ============================================================================
//  Frame access
// ============================================================================

/**
 * CG_GetFrame
 * Copies the latest grabbed frame into a caller-supplied buffer.
 * Set out_frame->data = NULL to query dimensions without copying pixels.
 */
CG_API CG_Status CG_GetFrame(CG_Frame* out_frame);

// ============================================================================
//  Frame saving
// ============================================================================

/**
 * CG_SaveFrame
 * Saves the current frame to  <work_dir>\images\frame_YYYYMMDD_HHMMSS.png
 * @param[out] out_path  Optional — receives the full saved path (MAX_PATH bytes).
 */
CG_API CG_Status CG_SaveFrame(char* out_path);

/**
 * CG_SaveFrameToPath
 * Saves the current frame to a caller-specified path.
 */
CG_API CG_Status CG_SaveFrameToPath(const char* path);

/**
 * CG_SaveRawFrame
 * Saves a CG_SaveFrame_roi - it saves image inside the ROI
 */
    
CG_API CG_Status CG_SaveFrame_roi(char*   out_path,
                                   int32_t roi_x, int32_t roi_y,
                                   int32_t roi_w, int32_t roi_h);
  
/**
 * CG_SaveRawFrame
 * Saves a CG_Frame (e.g. from a callback) to disk without needing the stream.
 */


CG_API CG_Status CG_SaveRawFrame(const CG_Frame* frame, const char* path);

// ============================================================================
//  Board presence
// ============================================================================

/**
 * CG_GetBoardPresence
 * Lightweight OpenCV board detector on the latest frame. No Python, no I/O.
 */
CG_API CG_Status CG_GetBoardPresence(CG_BoardPresence* out);

// ============================================================================
//  ROI management  (v4 addition)
// ============================================================================

/**
 * CG_SetROI
 * Store the IC bounding-box used by CG_SetReference and CG_InspectIC.
 * Call once after the user draws the region of interest on screen.
 * Pass roi_w = roi_h = 0 to clear (full-frame mode).
 * Changing the ROI automatically invalidates the existing reference.
 */
CG_API CG_Status CG_SetROI(int32_t roi_x, int32_t roi_y,
                             int32_t roi_w, int32_t roi_h);

// ============================================================================
//  Reference management
// ============================================================================

/**
 * CG_SetReference
 * Grabs the latest live frame, crops it to the stored ROI, runs YOLO via
 * Python (--mode check_roi) to confirm IC presence, then saves the crop as
 * ref\current_reference.png.
 * Returns CG_ERR_NO_BOARD if YOLO finds no IC — caller must retry.
 */
CG_API CG_Status CG_SetReference(void);

CG_API CG_Status CG_LoadReferenceFromFile(const char* image_path,
                                           int32_t roi_x, int32_t roi_y,
                                           int32_t roi_w, int32_t roi_h);
 
/**
 * CG_SetReference
 *   Grabs the latest live frame, crops to roi, YOLO-checks IC presence,
 *   and saves the crop as ref\current_reference.png.
 *
 *   roi_x/y/w/h  — ROI in full-frame pixel coordinates (0,0,0,0 → whole frame)
 */

/** Load an existing image from disk as the reference (no YOLO check). */
CG_API CG_Status CG_SetReferenceFromPath(const char* path);

/**
 * CG_SetReferenceFromFrame  (v4 addition)
 * Same as CG_SetReference but operates on a caller-supplied CG_Frame.
 * Runs YOLO presence check before accepting the frame as the reference.
 * Designed for use inside a CG_FrameCallback.
 *
 * @param frame       Pointer to a valid CG_Frame (data != NULL).
 * @param roi_x/y/w/h IC bounding box in frame pixels.
 *                    Pass roi_w = roi_h = 0 to use the full frame.
 * @return CG_OK, CG_ERR_INVALID_ARG, CG_ERR_NO_BOARD, CG_ERR_SAVE_FAILED.
 */
CG_API CG_Status CG_SetReferenceFromFrame(const CG_Frame* frame,
                                           int32_t roi_x, int32_t roi_y,
                                           int32_t roi_w, int32_t roi_h);

// ============================================================================
//  IC Inspection  — DLL lightweight stub (no Python)
// ============================================================================

/**
 * CG_InspectIC
 *
 * Full three-stage IC inspection on the live stream (v4: was a stub).
 *   1. Grabs the latest frame and crops it to the supplied ROI.
 *   2. Saves the crop as images\current_test.png.
 *   3. Runs Python integrated_detector.py --mode interactive.
 *   4. Parses output → fills *out (CG_DetectionResult summary).
 *      The rich CG_ICResult is available via CG_GetLastICResult().
 *
 * Returns CG_ERR_NO_REFERENCE if CG_SetReference has not been called yet.
 *
 * @param roi_x / roi_y / roi_w / roi_h
 *           Bounding box of the IC in full-frame pixel coordinates.
 * @param expected_dot_quadrant
 *           Where the polarity dot should appear:
 *             0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
 * @param[out] out  Caller-allocated CG_DetectionResult.
 * @return CG_OK, CG_ERR_STREAM_INACTIVE, CG_ERR_NO_REFERENCE,
 *         CG_ERR_GRAB_FAILED, CG_ERR_PYTHON_FAILED, CG_ERR_INVALID_ARG.
 */
CG_API CG_Status CG_InspectIC(int32_t roi_x, int32_t roi_y,
                                int32_t roi_w, int32_t roi_h,
                                int32_t expected_dot_quadrant,
                                CG_DetectionResult* out);

/**
 * CG_InspectICFromFrame
 *
 * Same full pipeline as CG_InspectIC but operates on a caller-supplied
 * CG_Frame instead of the live stream. Designed for use inside a
 * CG_FrameCallback.
 *
 * Returns CG_ERR_NO_REFERENCE if CG_SetReference has not been called yet.
 *
 * @param frame                  Pointer to a valid CG_Frame (data != NULL).
 * @param roi_x/y/w/h            Bounding box in frame pixels.
 * @param expected_dot_quadrant  Expected polarity dot quadrant (0-3).
 * @param[out] out               Caller-allocated CG_DetectionResult.
 * @return CG_OK, CG_ERR_NO_REFERENCE, CG_ERR_PYTHON_FAILED, CG_ERR_INVALID_ARG.
 */
CG_API CG_Status CG_InspectICFromFrame(const CG_Frame*     frame,
                                         int32_t             roi_x,
                                         int32_t             roi_y,
                                         int32_t             roi_w,
                                         int32_t             roi_h,
                                         int32_t             expected_dot_quadrant,
                                         CG_DetectionResult* out);

/**
 * CG_GetLastICResult  (v4 addition)
 * Returns the full CG_ICResult from the most recent CG_InspectIC /
 * CG_InspectICFromFrame call.  Contains pin counts, rotation angle,
 * orientation step, and all sub-fault counts.
 * @return CG_OK if a result exists; CG_ERR_GRAB_FAILED if no inspection
 *         has been performed yet.
 */
CG_API CG_Status CG_GetLastICResult(CG_ICResult* out);


CG_API const char* CG_GetLastError(void);

/** Fill *major / *minor / *patch with the DLL version. */
CG_API void CG_GetVersion(int* major, int* minor, int* patch);

/** Fill buf (size bytes) with the work directory used for all relative paths. */
CG_API void CG_GetWorkDir(char* buf, int size);

#ifdef __cplusplus
}   // extern "C"
#endif