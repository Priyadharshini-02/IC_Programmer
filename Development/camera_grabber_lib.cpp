/*
 * camera_grabber_lib.cpp  —  DLL implementation for camera_grabber_api.h  (v4)
 *
 * Python-calling pipeline ported verbatim from camera_grabber_old.cpp:
 *   checkICPresenceInROI  →  called inside CG_SetReference / CG_SetReferenceFromFrame
 *   runFullInspection     →  called inside CG_InspectIC   / CG_InspectICFromFrame
 *
 * The main() loop, UI, and keyboard handling stay in camera_grabber.cpp.
 * This DLL exposes only the functions declared in camera_grabber_api.h.
 *
 * Build (MSVC example):
 *   cl /DBUILDING_DLL /EHsc /MD /LD camera_grabber_lib.cpp
 *      /I "$(PYLON_ROOT)\include" /I "$(OPENCV_ROOT)\include"
 *      /link "$(PYLON_ROOT)\lib\x64\*.lib" opencv_world4xx.lib
 */

#define BUILDING_DLL
#define NOMINMAX

#include "camera_grabber_api.h"

// ── Pylon ────────────────────────────────────────────────────────────────────
#include <pylon/PylonIncludes.h>

// ── OpenCV ───────────────────────────────────────────────────────────────────
#include <opencv2/opencv.hpp>

// ── Windows / CRT ────────────────────────────────────────────────────────────
#include <windows.h>
#include <direct.h>
#include <io.h>

// ── Standard library ─────────────────────────────────────────────────────────
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

// =============================================================================
//  DllMain — capture HMODULE so we can locate the DLL directory at runtime
// =============================================================================
static HMODULE g_hmodule = nullptr;

BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_hmodule = hInst;
        DisableThreadLibraryCalls(hInst);
    }
    return TRUE;
}

// =============================================================================
//  Version
// =============================================================================
static constexpr int VER_MAJOR = 4, VER_MINOR = 0, VER_PATCH = 0;

// =============================================================================
//  FOV — define the live-stream field of view here (post-rotation coordinates).
//  These are applied automatically when CG_StartStream() is called.
//  Set FOV_W / FOV_H to 0 to stream the full frame.
// =============================================================================
static constexpr int FOV_X = 600;    // left edge  (pixels, post-rotation)
static constexpr int FOV_Y = 600;    // top  edge  (pixels, post-rotation)
static constexpr int FOV_W = 1200; // width      — change to your required FOV
static constexpr int FOV_H = 900;  // height     — change to your required FOV

// =============================================================================
//  Global state
// =============================================================================

// ── Lifecycle ─────────────────────────────────────────────────────────────────
static atomic<bool>  g_initialized{false};
static atomic<bool>  g_streaming  {false};

// ── Camera ────────────────────────────────────────────────────────────────────
static Pylon::CInstantCamera         g_camera;
static Pylon::CImageFormatConverter  g_converter;

// ── Latest grabbed frame  (BGR, DLL-owned buffer) ─────────────────────────────
static mutex            g_frame_mutex;
static vector<uint8_t>  g_frame_buf;
static CG_Frame         g_last_frame{};
static uint32_t         g_frame_index{0};

// ── User frame callback ───────────────────────────────────────────────────────
static mutex             g_cb_mutex;
static CG_FrameCallback  g_callback      = nullptr;
static void*             g_callback_user = nullptr;

// ── Grab thread ───────────────────────────────────────────────────────────────
static thread g_grab_thread;

// ── Last error ────────────────────────────────────────────────────────────────
static mutex  g_err_mutex;
static string g_last_error_str;

// ── Work directory  (set to DLL folder in CG_Initialize) ──────────────────────
//    Python scripts, ref/, images/, tmp/, output/ all live here.
static string g_work_dir;

// ── ROI  (full-frame pixel coordinates, set via CG_SetROI) ────────────────────
static mutex  g_roi_mutex;
static Rect   g_roi_rect;
static bool   g_has_roi{false};

// ── Crop  (applied to every grabbed frame after rotation, caller-controlled) ──
static mutex  g_crop_mutex;
static Rect   g_crop_rect;
static bool   g_has_crop{false};

// ── Reference ─────────────────────────────────────────────────────────────────
static mutex  g_ref_mutex;
static bool   g_has_reference{false};
static string g_ref_path;          // relative to g_work_dir; passed to Python

// ── Last rich IC result ───────────────────────────────────────────────────────
static mutex       g_result_mutex;
static CG_ICResult g_last_ic_result{};
static bool        g_has_ic_result{false};

// =============================================================================
//  Internal helpers
// =============================================================================

// Absolute path to a file inside the work directory
static string P(const string& sub)
{
    return g_work_dir.empty() ? sub : g_work_dir + "\\" + sub;
}

static void setError(const string& msg)
{
    lock_guard<mutex> lk(g_err_mutex);
    g_last_error_str = msg;
}

static void createDir(const string& path)
{
    CreateDirectoryA(path.c_str(), nullptr);
}

static bool fileExists(const string& path)
{
    return _access(path.c_str(), 0) == 0;
}

static string makeTimestamp()
{
    auto now = chrono::system_clock::now();
    auto t   = chrono::system_clock::to_time_t(now);
    char ts[32];
    struct tm tb{}; localtime_s(&tb, &t);
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tb);
    return string(ts);
}

// Thread-safe copy of the latest frame into a caller-owned buffer
static CG_Status snapshotFrame(CG_Frame& out_frame, vector<uint8_t>& out_buf)
{
    lock_guard<mutex> lk(g_frame_mutex);
    if (!g_last_frame.width || !g_last_frame.height || !g_last_frame.data)
    {
        setError("snapshotFrame: no valid frame in buffer");
        return CG_ERR_GRAB_FAILED;
    }
    size_t sz = static_cast<size_t>(g_last_frame.step) * g_last_frame.height;
    out_buf.resize(sz);
    memcpy(out_buf.data(), g_last_frame.data, sz);
    out_frame      = g_last_frame;
    out_frame.data = out_buf.data();
    return CG_OK;
}

// Write ref\roi_coords.txt consumed by Python --roi-file
static bool writeROIFile(const Rect& roi)
{
    ofstream f(P("ref\\roi_coords.txt"));
    if (!f.is_open()) { setError("Cannot write ref\\roi_coords.txt"); return false; }
    f << roi.x << " " << roi.y << " "
      << (roi.x + roi.width) << " " << (roi.y + roi.height) << "\n";
    return true;
}

// =============================================================================
//  Python output parser helpers  — identical to camera_grabber_old.cpp
// =============================================================================
static const char* skipTokenAndSpaces(const string& ln, const string& tok)
{
    size_t p = ln.find(tok);
    if (p == string::npos) return nullptr;
    const char* s = ln.c_str() + p + tok.size();
    while (*s == ' ' || *s == ':') ++s;
    return s;
}

static int parseAfterInt(const string& ln, const string& tok)
{
    const char* s = skipTokenAndSpaces(ln, tok);
    if (!s) return -1;
    try { return stoi(s); } catch (...) { return -1; }
}

static float parseAfterFloat(const string& ln, const string& tok)
{
    const char* s = skipTokenAndSpaces(ln, tok);
    if (!s) return -999.f;
    try { return stof(s); } catch (...) { return -999.f; }
}

// =============================================================================
//  Internal: YOLO presence check
//  Returns one of three outcomes so callers can emit precise error messages:
//    IC_CHECK_PRESENT      — IC detected AND its centre is inside the ROI
//    IC_CHECK_ABSENT       — YOLO found no IC anywhere in the frame
//    IC_CHECK_OUTSIDE_ROI  — YOLO found an IC but it is outside the ROI
//    IC_CHECK_ERROR        — subprocess / I-O failure (error string already set)
// =============================================================================
enum ICCheckResult { IC_CHECK_PRESENT, IC_CHECK_ABSENT, IC_CHECK_OUTSIDE_ROI, IC_CHECK_ERROR };

static ICCheckResult checkICPresenceInROI(const CG_Frame& cgf, const Rect& roi)
{
    // ── 1. Save frame to disk so Python can read it ───────────────────────────
    const string img_path = P("images\\roi_check.png");
    if (CG_SaveRawFrame(&cgf, img_path.c_str()) != CG_OK)
    {
        setError("[Presence] CG_SaveRawFrame failed");
        return IC_CHECK_ERROR;
    }

    // ── 2. Write ROI as x1 y1 x2 y2  (Python reads it the same way) ──────────
    {
        ofstream f(P("tmp\\roi_coords.txt"));
        if (!f.is_open())
        {
            setError("[Presence] Cannot write tmp\\roi_coords.txt");
            return IC_CHECK_ERROR;
        }
        f << roi.x              << " " << roi.y
          << " " << (roi.x + roi.width)
          << " " << (roi.y + roi.height) << "\n";
    }

    // ── 3. Log what we are about to run so failures are reproducible ─────────
    printf("[Presence] ROI: x1=%d y1=%d x2=%d y2=%d\n",
           roi.x, roi.y, roi.x + roi.width, roi.y + roi.height);
    printf("[Presence] Image: %s\n", img_path.c_str());
    fflush(stdout);

    // ── 4. Spawn Python:
    //        --mode check_roi  → runs YOLO on the full image,
    //                            checks whether any detected IC centre
    //                            falls inside the ROI,
    //                            prints IC_PRESENT or IC_ABSENT ────────────────
    const string result_path = P("tmp\\roi_check_result.txt");
    const string cmd =
        "cd /d \"" + g_work_dir + "\" && "
        "python integrated_detector.py "
        "--mode check_roi "
        "--image \"images\\roi_check.png\" "
        "--roi-file \"tmp\\roi_coords.txt\" "
        "> \"" + result_path + "\" 2>&1";

    printf("[Presence] Running: %s\n", cmd.c_str());
    fflush(stdout);

    int rc = system(cmd.c_str());

    // ── 5. Always read and print the Python output — even on failure ──────────
    //       This is the key diagnostic: it shows YOLO load errors, import
    //       errors, model-not-found, detected count, ROI miss reason, etc.
    {
        ifstream dbg(result_path);
        if (dbg.is_open())
        {
            printf("[Presence] ── Python output ────────────────────────────────\n");
            string dl;
            while (getline(dbg, dl))
                printf("  %s\n", dl.c_str());
            printf("[Presence] ────────────────────────────────────────────────\n");
            fflush(stdout);
        }
        else
        {
            printf("[Presence] WARNING: result file missing — Python may not have started\n");
            fflush(stdout);
        }
    }

    if (rc != 0)
    {
        setError("[Presence] Python check_roi subprocess returned non-zero — see console output above");
        return IC_CHECK_ERROR;
    }

    // ── 6. Parse result — IC_PRESENT / IC_ABSENT may have a comment suffix ────
    ifstream f(result_path);
    if (!f.is_open())
    {
        setError("[Presence] Result file not readable after successful subprocess");
        return IC_CHECK_ERROR;
    }

    string line;
    while (getline(f, line))
    {
        if (line.find("IC_PRESENT")     != string::npos) return IC_CHECK_PRESENT;
        if (line.find("IC_OUTSIDE_ROI") != string::npos) return IC_CHECK_OUTSIDE_ROI;
        if (line.find("IC_ABSENT")      != string::npos) return IC_CHECK_ABSENT;
    }

    setError("[Presence] No IC_PRESENT/IC_ABSENT/IC_OUTSIDE_ROI token in Python output — see console");
    return IC_CHECK_ERROR;
}

// =============================================================================
//  Internal: run Python set_reference so ROI-cropped chip cache is generated
//  Called after every CG_SetReference / CG_SetReferenceFromFrame /
//  CG_LoadReferenceFromFile so that ref/chip_N.png and ref/ref_meta.json
//  always contain chips cropped to the current ROI.
// =============================================================================
static CG_Status runReferenceSetup(const string& ref_img_path,
                                   int32_t roi_x = 0, int32_t roi_y = 0,
                                   int32_t roi_w = 0, int32_t roi_h = 0)
{
    const string input_path  = P("tmp\\ref_setup_input.txt");
    const string output_path = P("tmp\\ref_setup_output.txt");

    // ── 1. Write Python input ───────────────────────────────────────
    {
        ofstream inp(input_path);
        if (!inp.is_open())
        {
            setError("runReferenceSetup: cannot write input file");
            return CG_ERR_SAVE_FAILED;
        }

        inp << "ref " << ref_img_path << "\n"
            << "quit\n";
    }

    // ── 2. Ensure ROI file exists (if ROI provided) ─────────────────
    if (roi_w > 0 && roi_h > 0)
    {
        ofstream rf(P("ref\\roi_coords.txt"));
        if (!rf.is_open())
        {
            setError("runReferenceSetup: cannot write ROI file");
            return CG_ERR_SAVE_FAILED;
        }

        rf << roi_x << " " << roi_y << " "
           << (roi_x + roi_w) << " " << (roi_y + roi_h) << "\n";
    }

    // ── 3. Run Python ───────────────────────────────────────────────
    const string cmd =
        "cd /d \"" + g_work_dir + "\" && "
        "python integrated_detector.py "
        "--mode interactive "
        "--roi-file \"ref\\roi_coords.txt\" "
        "< tmp\\ref_setup_input.txt "
        "> tmp\\ref_setup_output.txt 2>&1";

    int ret = system(cmd.c_str());

    // ── 4. Check output file exists ─────────────────────────────────
    if (!fileExists(output_path))
    {
        setError("runReferenceSetup: Python did not produce output file");
        return CG_ERR_PYTHON_FAILED;
    }

    // ── 5. Check if chip was created (actual success condition) ─────
    const string chip0 = P("ref\\chip_0.png");

    if (!fileExists(chip0))
    {
        // Optional: read Python output for debugging
        ifstream f(output_path);
        string log((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        fprintf(stderr, "[Python Output]\n%s\n", log.c_str());

        setError("runReferenceSetup: no IC found inside ROI");
        return CG_ERR_NO_BOARD;
    }

    // ── 6. Success ──────────────────────────────────────────────────
    return CG_OK;
}


// =============================================================================
//  Internal: full Python inspection
//  
// =============================================================================
static CG_Status runFullInspection(const CG_Frame& cgf,
                                    const string&   ref_path,
                                    CG_ICResult*    out,
                                    int32_t roi_x = 0, int32_t roi_y = 0,
                                    int32_t roi_w = 0, int32_t roi_h = 0)
{
    if (!out) return CG_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->status = CG_IC_ERROR;

    // Save test frame — zero-copy Mat wrapper, no clone needed for imwrite
    const string test_img_path = P("images\\current_test.png");
    {
        Mat test_mat(cgf.height, cgf.width, CV_8UC3,
                     const_cast<uint8_t*>(cgf.data),
                     static_cast<size_t>(cgf.step));
        if (test_mat.empty() || !imwrite(test_img_path, test_mat))
        {
            snprintf(out->message, sizeof(out->message), "imwrite failed for test frame");
            setError(out->message);
            return CG_ERR_SAVE_FAILED;
        }
    }

    // Write stdin commands for Python interactive mode.
    // load_ref <ref_path>  — loads the pre-cropped chip without re-running YOLO.
    // test <test_path>     — detects IC in the full test frame, ROI-filters,
    //                        crops the matching chip, and compares against the
    //                        loaded reference chip (orientation + pin health).
    const string input_path  = P("tmp\\detector_input.txt");
    const string output_path = P("tmp\\inspection_output.txt");
    {
        ofstream inp(input_path);
        if (!inp.is_open())
        {
            snprintf(out->message, sizeof(out->message), "Cannot write detector_input.txt");
            setError(out->message);
            return CG_ERR_SAVE_FAILED;
        }
        inp << "load_ref " << ref_path << "\n"
            << "test "     << "images\\current_test.png" << "\n"
            << "quit\n";
    }

    // Write test-time ROI so Python filters detections to the correct region.
    // If no ROI is supplied fall back to the reference-time roi_coords.txt.
    const bool   has_test_roi   = (roi_w > 0 && roi_h > 0);
    const string test_roi_path  = P("tmp\\test_roi_coords.txt");
    const string roi_file_arg   = has_test_roi
                                    ? "tmp\\test_roi_coords.txt"
                                    : "ref\\roi_coords.txt";
    if (has_test_roi)
    {
        ofstream rf(test_roi_path);
        if (!rf.is_open())
        {
            snprintf(out->message, sizeof(out->message),
                     "Cannot write tmp\\test_roi_coords.txt");
            setError(out->message);
            return CG_ERR_SAVE_FAILED;
        }
        // Python --roi-file expects: x1 y1 x2 y2
        rf << roi_x << " " << roi_y << " "
           << (roi_x + roi_w) << " " << (roi_y + roi_h) << "\n";
    }

    // Launch Python — system() blocks until Python exits, so no sleep needed
    const string cmd =
        "cd /d \"" + g_work_dir + "\" && "
        "python integrated_detector.py "
        "--mode interactive "
        "--roi-file \"" + roi_file_arg + "\" "
        "< tmp\\detector_input.txt "
        "> tmp\\inspection_output.txt 2>&1";

    if (system(cmd.c_str()) != 0)
    {
        snprintf(out->message, sizeof(out->message), "Python subprocess failed");
        setError(out->message);
        return CG_ERR_PYTHON_FAILED;
    }

    // Parse output — only lines inside the [RESULT_BEGIN]..[RESULT_END] block.
    // Python emits the same token strings twice: once in a human-readable
    // section (with Unicode and spaces) and once in the machine-readable block.
    // Restricting to the machine-readable block avoids ambiguous substring
    // matches on the human-readable lines regardless of output ordering.
    ifstream pyout(output_path);
    if (!pyout.is_open())
    {
        snprintf(out->message, sizeof(out->message), "Python output file not found");
        setError(out->message);
        return CG_ERR_PARSE_FAILED;
    }

    string line;
    bool   in_result_block = false;
    while (getline(pyout, line))
    {
        if (line.find("[RESULT_BEGIN]") != string::npos) { in_result_block = true;  continue; }
        if (line.find("[RESULT_END]")   != string::npos) { in_result_block = false; continue; }
        if (!in_result_block) continue;

        int   iv;
        float fv;

        if ((iv = parseAfterInt(line,   "IC Present (matched):"))    >= 0) out->present_count           = iv;
        if ((iv = parseAfterInt(line,   "IC Absent  (missing):"))    >= 0) out->absent_count            = iv;
        if ((iv = parseAfterInt(line,   "IC Extra   (unexpected):")) >= 0) out->extra_count             = iv;
        if ((iv = parseAfterInt(line,   "Wrong orientation:"))       >= 0) out->wrong_orientation_count = iv;
        if ((iv = parseAfterInt(line,   "Pin health failures:"))     >= 0) out->pin_fail_count          = iv;
        if ((iv = parseAfterInt(line,   "pin_missing="))             >= 0) out->pin_missing             = iv;
        if ((iv = parseAfterInt(line,   "pin_bent="))                >= 0) out->pin_bent                = iv;
        if ((iv = parseAfterInt(line,   "pin_bridge="))              >= 0) out->pin_bridged             = iv;

        if ((fv = parseAfterFloat(line, "rotation=")) > -999.f)
        {
            out->rotation_deg = fv;
            size_t sp = line.find("step~");
            if (sp != string::npos)
            {
                try { out->rotation_step = stoi(line.substr(sp + 5)); } catch (...) {}
            }
        }
    }

    // Map parsed counts → CG_ICStatus  (priority: absent → wrong_orientation → pin → extra → pass)
    if (out->absent_count > 0)
    {
        out->status = CG_IC_ABSENT;
        snprintf(out->message, sizeof(out->message),
                 "IC absent: %d missing", out->absent_count);
    }
    else if (out->wrong_orientation_count > 0)
    {
        out->status = CG_IC_WRONG_ORIENTATION;
        snprintf(out->message, sizeof(out->message),
                 "Wrong orientation: rotation=%.1fdeg step=%ddeg",
                 out->rotation_deg, out->rotation_step);
    }
    else if (out->pin_fail_count > 0)
    {
        out->status = CG_IC_PIN_DAMAGED;
        snprintf(out->message, sizeof(out->message),
                 "Pin damage: miss=%d bent=%d bridged=%d",
                 out->pin_missing, out->pin_bent, out->pin_bridged);
    }
    else if (out->extra_count > 0)
    {
        out->status = CG_IC_EXTRA;
        snprintf(out->message, sizeof(out->message),
                 "Extra IC: %d unexpected", out->extra_count);
    }
    else
    {
        out->status = CG_IC_PASS;
        snprintf(out->message, sizeof(out->message), "PASS");
    }

    return CG_OK;
}

// =============================================================================
//  Grab thread
// =============================================================================
static void grabThreadFunc()
{
    g_converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;

    while (g_streaming.load())
    {
        try
        {
            Pylon::CGrabResultPtr ptrResult;
            if (!g_camera.RetrieveResult(500,
                                         ptrResult,
                                         Pylon::TimeoutHandling_Return))
                continue;

            if (!ptrResult || !ptrResult->GrabSucceeded())
                continue;

            Pylon::CPylonImage pylonImage;
g_converter.Convert(pylonImage, ptrResult);

// ── Rotate 90° clockwise ─────────────────────────────────────────────────────
const int src_w = static_cast<int>(ptrResult->GetWidth());
const int src_h = static_cast<int>(ptrResult->GetHeight());

Mat raw(src_h, src_w, CV_8UC3,
        pylonImage.GetBuffer(),
        static_cast<size_t>(src_w) * 3);   // zero-copy wrapper

Mat rotated;
cv::rotate(raw, rotated, cv::ROTATE_90_CLOCKWISE); // allocates its own buffer

// ── FOV crop (applied after rotation — stream only delivers this region) ──────
//    Set via CG_SetFOV(); cleared via CG_ClearFOV().
//    When active every downstream consumer (g_last_frame, callback, snapshots,
//    inspection) sees only the cropped region — the full sensor frame is never
//    exposed outside the grab thread.
{
    lock_guard<mutex> lk(g_crop_mutex);
    if (g_has_crop)
    {
        // Clamp to actual frame size so an out-of-bounds FOV never asserts
        Rect safe = g_crop_rect & Rect(0, 0, rotated.cols, rotated.rows);
        if (safe.width > 0 && safe.height > 0)
            rotated = rotated(safe).clone(); // clone → contiguous owned buffer
    }
}

// After rotation + FOV crop: cols/rows are the final delivered dimensions
const int    w    = rotated.cols;
const int    h    = rotated.rows;
const int    step = static_cast<int>(rotated.step);
const size_t sz   = static_cast<size_t>(step) * h;

// ── Update shared frame buffer ───────────────────────────────────────────────
{
    lock_guard<mutex> lk(g_frame_mutex);
    g_frame_buf.resize(sz);
    memcpy(g_frame_buf.data(), rotated.data, sz);  // ← rotated.data, NOT pylonImage

    g_last_frame.data         = g_frame_buf.data();
    g_last_frame.width        = w;
    g_last_frame.height       = h;
    g_last_frame.step         = step;
    g_last_frame.timestamp_ms =
        static_cast<int64_t>(ptrResult->GetTimeStamp() / 1'000'000);
    g_last_frame.frame_index  = ++g_frame_index;
}

// ── Fire user callback ───────────────────────────────────────────────────────
CG_FrameCallback cb   = nullptr;
void*            user = nullptr;
{
    lock_guard<mutex> lk(g_cb_mutex);
    cb   = g_callback;
    user = g_callback_user;
}
if (cb)
{
    CG_Frame cbf      = g_last_frame;
    cbf.data          = rotated.data;   // ← rotated.data, NOT pylonImage.GetBuffer()
    cb(&cbf, user);
}
        }
        catch (const Pylon::GenericException& e)
        {
            setError(string("Grab error: ") + e.GetDescription());
        }
        catch (const exception& e)
        {
            setError(string("Grab error: ") + e.what());
        }
    }
}

// =============================================================================
//  Shared inspection core  (used by both CG_InspectIC and CG_InspectICFromFrame)
// =============================================================================
static CG_Status inspectFrame(const CG_Frame& cgf, CG_DetectionResult* out,
                               int32_t roi_x = 0, int32_t roi_y = 0,
                               int32_t roi_w = 0, int32_t roi_h = 0)
{
    if (!out) return CG_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    string ref_path;
    {
        lock_guard<mutex> lk(g_ref_mutex);
        if (!g_has_reference)
        {
            setError("No reference set — call CG_SetReference first");
            return CG_ERR_NO_REFERENCE;
        }
        ref_path = g_ref_path;
    }

    CG_ICResult rich{};
    CG_Status st = runFullInspection(cgf, ref_path, &rich,
                                      roi_x, roi_y, roi_w, roi_h);

    if (st == CG_OK)
    {
        // Populate lightweight CG_DetectionResult from the parsed rich result
        out->present =
            (rich.status == CG_IC_PASS            ||
             rich.status == CG_IC_WRONG_ORIENTATION ||
             rich.status == CG_IC_PIN_DAMAGED) ? 1 : 0;
        out->pindamage         = (rich.status == CG_IC_PIN_DAMAGED)       ? 1 : 0;
        out->wrong_orientation = (rich.status == CG_IC_WRONG_ORIENTATION) ? 1 : 0;

        // Single summary component entry
        CG_Component& c = out->components[0];
        c.id    = 0;
        c.score = out->present ? 1.0f : 0.0f;
        strncpy_s(c.cls,    sizeof(c.cls),    "IC",          _TRUNCATE);
        strncpy_s(c.status, sizeof(c.status), rich.message,  _TRUNCATE);

        // Persist rich result for CG_GetLastICResult
        {
            lock_guard<mutex> lk(g_result_mutex);
            g_last_ic_result = rich;
            g_has_ic_result  = true;
        }
    }
    return st;
}

// =============================================================================
//  Public API
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
//  Lifecycle
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_Initialize(void)
{
    if (g_initialized.exchange(true))
    {
        setError("CG_Initialize: already initialised");
        return CG_ERR_ALREADY_INIT;
    }

    // Resolve DLL directory → use as work dir so Python scripts are found
    {
        char dll_path[MAX_PATH] = {};
        GetModuleFileNameA(g_hmodule, dll_path, MAX_PATH);
        string s(dll_path);
        size_t pos = s.rfind('\\');
        g_work_dir = (pos != string::npos) ? s.substr(0, pos) : ".";
    }

    // Create all required directories once at startup
    createDir(P("ref"));
    createDir(P("images"));
    createDir(P("tmp"));
    createDir(P("output"));

    Pylon::PylonInitialize();
    return CG_OK;
}

CG_API void CG_Shutdown(void)
{
    CG_StopStream();
    if (g_initialized.load())
    {
        Pylon::PylonTerminate();
        g_initialized.store(false);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Streaming
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_StartStream(void)
{
    if (!g_initialized.load()) return CG_ERR_NOT_INIT;
    if (g_streaming.load())    return CG_ERR_STREAM_ACTIVE;

    try
    {
        g_camera.Attach(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
        g_camera.Open();
        g_camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly,
                                Pylon::GrabLoop_ProvidedByUser);
    }
    catch (const Pylon::GenericException& e)
    {
        setError(string("CG_StartStream: ") + e.GetDescription());
        return CG_ERR_NO_CAMERA;
    }

    g_streaming.store(true);

    // ── Apply built-in FOV so the stream is locked to the defined region ──────
    if (FOV_W > 0 && FOV_H > 0)
    {
        lock_guard<mutex> lk(g_crop_mutex);
        g_crop_rect = Rect(FOV_X, FOV_Y, FOV_W, FOV_H);
        g_has_crop  = true;
        printf("[CG_StartStream] FOV active: x=%d y=%d w=%d h=%d\n",
               FOV_X, FOV_Y, FOV_W, FOV_H);
        fflush(stdout);
    }

    g_grab_thread = thread(grabThreadFunc);
    return CG_OK;
}

CG_API CG_Status CG_StopStream(void)
{
    if (!g_streaming.load()) return CG_OK;   // safe to call when not streaming

    g_streaming.store(false);
    if (g_grab_thread.joinable())
        g_grab_thread.join();

    try
    {
        if (g_camera.IsGrabbing())            g_camera.StopGrabbing();
        if (g_camera.IsOpen())                g_camera.Close();
        if (g_camera.IsPylonDeviceAttached()) g_camera.DetachDevice();
    }
    catch (const Pylon::GenericException&) {}

    return CG_OK;
}

CG_API int CG_IsStreaming(void)
{
    return g_streaming.load() ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Callback
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_SetFrameCallback(CG_FrameCallback cb, void* user)
{
    lock_guard<mutex> lk(g_cb_mutex);
    g_callback      = cb;
    g_callback_user = user;
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frame access
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_GetFrame(CG_Frame* out_frame)
{
    if (!out_frame)             return CG_ERR_INVALID_ARG;
    if (!g_streaming.load())   return CG_ERR_STREAM_INACTIVE;

    lock_guard<mutex> lk(g_frame_mutex);
    if (!g_last_frame.width || !g_last_frame.data)
    {
        setError("CG_GetFrame: no frame in buffer yet");
        return CG_ERR_GRAB_FAILED;
    }

    // Dimension-query shortcut (caller passes data == NULL)
    if (out_frame->data == nullptr)
    {
        out_frame->width        = g_last_frame.width;
        out_frame->height       = g_last_frame.height;
        out_frame->step         = g_last_frame.step;
        out_frame->timestamp_ms = g_last_frame.timestamp_ms;
        out_frame->frame_index  = g_last_frame.frame_index;
        return CG_OK;
    }

    size_t sz = static_cast<size_t>(g_last_frame.step) * g_last_frame.height;
    memcpy(out_frame->data, g_last_frame.data, sz);
    out_frame->width        = g_last_frame.width;
    out_frame->height       = g_last_frame.height;
    out_frame->step         = g_last_frame.step;
    out_frame->timestamp_ms = g_last_frame.timestamp_ms;
    out_frame->frame_index  = g_last_frame.frame_index;
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frame saving
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_SaveFrame(char* out_path)
{
    if (!g_streaming.load()) return CG_ERR_STREAM_INACTIVE;

    CG_Frame snap{}; vector<uint8_t> buf;
    CG_Status st = snapshotFrame(snap, buf);
    if (st != CG_OK) return st;

    string path = P("images\\frame_" + makeTimestamp() + ".png");
    Mat m(snap.height, snap.width, CV_8UC3, snap.data, snap.step);
    if (!imwrite(path, m))
    {
        setError("CG_SaveFrame: imwrite failed");
        return CG_ERR_SAVE_FAILED;
    }
    if (out_path)
        strncpy_s(out_path, MAX_PATH, path.c_str(), _TRUNCATE);
    return CG_OK;
}

CG_API CG_Status CG_SaveFrameToPath(const char* path)
{
    if (!path)                   return CG_ERR_INVALID_ARG;
    if (!g_streaming.load())     return CG_ERR_STREAM_INACTIVE;

    CG_Frame snap{}; vector<uint8_t> buf;
    CG_Status st = snapshotFrame(snap, buf);
    if (st != CG_OK) return st;

    Mat m(snap.height, snap.width, CV_8UC3, snap.data, snap.step);
    if (!imwrite(path, m))
    {
        setError("CG_SaveFrameToPath: imwrite failed");
        return CG_ERR_SAVE_FAILED;
    }
    return CG_OK;
}

CG_API CG_Status CG_SaveFrame_roi(char*   out_path,
                                   int32_t roi_x, int32_t roi_y,
                                   int32_t roi_w, int32_t roi_h)
{
    if (roi_w <= 0 || roi_h <= 0)
    {
        setError("CG_SaveFrame_roi: invalid ROI dimensions");
        return CG_ERR_INVALID_ARG;
    }
    if (!g_streaming.load()) return CG_ERR_STREAM_INACTIVE;

    // Snapshot the full live frame into a DLL-owned buffer
    CG_Frame snap{}; vector<uint8_t> buf;
    CG_Status st = snapshotFrame(snap, buf);
    if (st != CG_OK) return st;

    Mat full(snap.height, snap.width, CV_8UC3, snap.data, snap.step);

    // Clamp the ROI so it never exceeds the actual frame dimensions
    Rect roi(roi_x, roi_y, roi_w, roi_h);
    roi &= Rect(0, 0, full.cols, full.rows);   // intersection with frame bounds

    if (roi.empty())
    {
        setError("CG_SaveFrame_roi: ROI does not intersect the frame");
        return CG_ERR_INVALID_ARG;
    }

    // Crop — zero-copy sub-matrix view, clone to get a contiguous buffer
    Mat cropped = full(roi).clone();

    // string path = P("images\\roi_" + makeTimestamp() + ".png");
    // string path =out_path;
    if (!imwrite(out_path, cropped))
    {
        setError("CG_SaveFrame_roi: imwrite failed");
        return CG_ERR_SAVE_FAILED;
    }

    // if (out_path)
    //     strncpy_s(out_path, MAX_PATH, path.c_str(), _TRUNCATE);

    return CG_OK;
}

CG_API CG_Status CG_SaveRawFrame(const CG_Frame* frame, const char* path)
{
    if (!frame || !path || !frame->data || !frame->width || !frame->height)
        return CG_ERR_INVALID_ARG;

    Mat m(frame->height, frame->width, CV_8UC3,
          const_cast<uint8_t*>(frame->data),
          static_cast<size_t>(frame->step));
    if (!imwrite(path, m))
    {
        setError(string("CG_SaveRawFrame: imwrite failed → ") + path);
        return CG_ERR_SAVE_FAILED;
    }
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Board presence  (lightweight OpenCV — no Python, no I/O)
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_GetBoardPresence(CG_BoardPresence* out)
{
    if (!out) return CG_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    CG_Frame snap{}; vector<uint8_t> buf;
    CG_Status st = snapshotFrame(snap, buf);
    if (st != CG_OK) return st;

    Mat bgr(snap.height, snap.width, CV_8UC3, snap.data, snap.step);
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);

    Scalar mean_v, std_v;
    meanStdDev(gray, mean_v, std_v);
    out->stddev = std_v[0];

    Mat edges;
    Canny(gray, edges, 50, 150);
    out->edge_ratio = static_cast<double>(countNonZero(edges)) /
                      static_cast<double>(gray.total());

    out->IC_present = (out->stddev > 20.0 && out->edge_ratio > 0.02) ? 1 : 0;
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ROI management
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_SetROI(int32_t roi_x, int32_t roi_y,
                             int32_t roi_w, int32_t roi_h)
{
    {
        lock_guard<mutex> lk(g_roi_mutex);
        if (roi_w <= 0 || roi_h <= 0)
        {
            g_roi_rect = Rect();
            g_has_roi  = false;
        }
        else
        {
            g_roi_rect = Rect(roi_x, roi_y, roi_w, roi_h);
            g_has_roi  = true;
        }
    }
    // Changing ROI invalidates the existing reference
    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_has_reference = false;
    }
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reference management
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_SetReference(void)
{
    // ── 1. Check Stream ─────────────────────────────────────────────
    if (!g_streaming.load())
        return CG_ERR_STREAM_INACTIVE;
 //Clear the ref folder
    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_has_reference = false;
        g_ref_path.clear();
    }


    // ── 2. Capture Current Frame ───────────────────────────────────
    CG_Frame snap{}; 
    vector<uint8_t> snap_buf;
    CG_Status st = snapshotFrame(snap, snap_buf);
    if (st != CG_OK) return st;

    // ── 3. Get ROI ─────────────────────────────────────────────────
    Rect roi; 
    bool has_roi;
    {
        lock_guard<mutex> lk(g_roi_mutex);
        roi     = g_roi_rect;
        has_roi = g_has_roi;
    }

    if (!has_roi)
    {
        setError("CG_SetReference: no ROI set");
        return CG_ERR_INVALID_ARG;
    }

    // ── 4. Save Frame (input for Python) ────────────────────────────
    const string raw_img = P("ref\\raw_frame.png");
    if (CG_SaveRawFrame(&snap, raw_img.c_str()) != CG_OK)
        return CG_ERR_SAVE_FAILED;

    // ── 5. Write ROI file (used by Python) ──────────────────────────
    if (!writeROIFile(roi))
        return CG_ERR_SAVE_FAILED;

    // ── 6. Detect IC + Validate ROI + Crop (Python handles all) ─────
    printf("[SetReference] Running detection + ROI validation...\n");
    fflush(stdout);

    CG_Status rst = runReferenceSetup(
        "ref\\raw_frame.png",
        roi.x, roi.y, roi.width, roi.height);

    if (rst != CG_OK)
    {
        setError("CG_SetReference: IC detection/ROI validation failed");
        return rst;
    }

    // ── 7. Save Reference Path ──────────────────────────────────────
    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_ref_path      = P("ref\\chip_0.png");
        g_has_reference = true;
    }

    printf("[SetReference] Reference created successfully.\n");

    return CG_OK;
}

CG_API CG_Status CG_SetReferenceFromFrame(const CG_Frame* frame,
                                           int32_t roi_x, int32_t roi_y,
                                           int32_t roi_w, int32_t roi_h)
{
    if (!frame || !frame->data) return CG_ERR_INVALID_ARG;

    bool has_roi = (roi_w > 0 && roi_h > 0);
    Rect roi(roi_x, roi_y, roi_w, roi_h);

    // Write ROI file before running Python so set_reference can filter by it
    if (has_roi)
    {
        {
            lock_guard<mutex> lk(g_roi_mutex);
            g_roi_rect = roi;
            g_has_roi  = true;
        }
        if (!writeROIFile(roi)) return CG_ERR_SAVE_FAILED;
    }

    // ── Step 1: save the raw frame for the presence check and setup
    const string raw_img = P("ref\\raw_frame.png");
    if (CG_SaveRawFrame(frame, raw_img.c_str()) != CG_OK)
        return CG_ERR_SAVE_FAILED;

    // ── Step 2: check IC presence and ROI membership before committing ──────
    if (!has_roi)
    {
        setError("CG_SetReferenceFromFrame: no ROI supplied — provide non-zero roi_w/roi_h");
        return CG_ERR_INVALID_ARG;
    }
    {
        printf("[SetReferenceFromFrame] Checking IC presence inside ROI...\n");
        fflush(stdout);
        ICCheckResult chk = checkICPresenceInROI(*frame, roi);
        switch (chk)
        {
            case IC_CHECK_PRESENT:
                printf("[SetReferenceFromFrame] IC detected inside ROI — proceeding.\n");
                break;
            case IC_CHECK_ABSENT:
                setError("CG_SetReferenceFromFrame: no IC detected anywhere in the frame");
                printf("[SetReferenceFromFrame] FAIL — no IC found in frame.\n");
                return CG_ERR_NO_BOARD;
            case IC_CHECK_OUTSIDE_ROI:
                setError("CG_SetReferenceFromFrame: IC detected but its centre is outside the ROI");
                printf("[SetReferenceFromFrame] FAIL — IC found but outside ROI.\n");
                return CG_ERR_NO_BOARD;
            case IC_CHECK_ERROR:
            default:
                printf("[SetReferenceFromFrame] FAIL — presence check subprocess error.\n");
                return CG_ERR_PYTHON_FAILED;
        }
    }

    // ── Step 3: run Python set_reference — YOLO + ROI crop + save chip ──────
    {
        CG_Status rst = runReferenceSetup(
            "ref\\raw_frame.png",
            roi_x, roi_y, roi_w, roi_h);
        if (rst != CG_OK) return rst;
    }

    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_ref_path      = P("ref\\chip_0.png");
        g_has_reference = true;
    }
    return CG_OK;
}

CG_API CG_Status CG_SetReferenceFromPath(const char* path)
{
    if (!path) return CG_ERR_INVALID_ARG;

    

    // ── Resolve to absolute path so Python can find it regardless of cwd ──────
    //    Python runs with  "cd /d g_work_dir", so a relative path from a
    //    different directory would silently fail.  GetFullPathNameA resolves
    //    relative paths against the process cwd at call time, which is correct.
    char abs_path[MAX_PATH] = {};
    if (!GetFullPathNameA(path, MAX_PATH, abs_path, nullptr))
    {
        setError(string("CG_SetReferenceFromPath: cannot resolve path → ") + path);
        return CG_ERR_INVALID_ARG;
    }

    // ── Verify the file actually exists before accepting it ───────────────────
    if (!fileExists(abs_path))
    {
        setError(string("CG_SetReferenceFromPath: file not found → ") + abs_path);
        return CG_ERR_INVALID_ARG;
    }

    // ── Write ref\roi_coords.txt from current ROI state ───────────────────────
    //    runFullInspection always passes --roi-file ref\roi_coords.txt to Python.
    //    CG_SetReferenceFromPath previously skipped this write, so Python would
    //    use a stale or missing coords file and compare the full frame instead
    //    of the ROI.  We write it here using whatever ROI is currently set.
    {
        Rect  roi;
        bool  has_roi;
        {
            lock_guard<mutex> lk(g_roi_mutex);
            roi     = g_roi_rect;
            has_roi = g_has_roi;
        }
        if (has_roi && !writeROIFile(roi))
            return CG_ERR_SAVE_FAILED;
    }

    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_ref_path      = abs_path;   // absolute — Python can open it from any cwd
        g_has_reference = true;
    }
    return CG_OK;
}

CG_API CG_Status CG_LoadReferenceFromFile(const char* image_path,
                                           int32_t roi_x, int32_t roi_y,
                                           int32_t roi_w, int32_t roi_h)
{
    (void)roi_x; (void)roi_y; (void)roi_w; (void)roi_h;


    if (!image_path)
    {
        setError("CG_LoadReferenceFromFile: null path");
        return CG_ERR_INVALID_ARG;
    }

    // Resolve to absolute path so Python can open it from any working directory
    char abs_path[MAX_PATH] = {};
    if (!GetFullPathNameA(image_path, MAX_PATH, abs_path, nullptr))
    {
        setError(string("CG_LoadReferenceFromFile: cannot resolve path → ") + image_path);
        return CG_ERR_INVALID_ARG;
    }

    if (!fileExists(abs_path))
    {
        setError(string("CG_LoadReferenceFromFile: file not found → ") + abs_path);
        return CG_ERR_INVALID_ARG;
    }

    // Store the chip path directly — no copy, no Python re-run needed.
    // runFullInspection will pass this path to Python via "load_ref <path>".
    {
        lock_guard<mutex> lk(g_ref_mutex);
        g_ref_path      = abs_path;
        g_has_reference = true;
    }
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  IC Inspection
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_InspectIC(int32_t roi_x, int32_t roi_y,
                                int32_t roi_w, int32_t roi_h,
                                int32_t expected_dot_quadrant,
                                CG_DetectionResult* out)
{
    (void)expected_dot_quadrant;   // handled by Python orientation logic
    if (!g_streaming.load()) return CG_ERR_STREAM_INACTIVE;

    CG_Frame snap{}; vector<uint8_t> snap_buf;
    CG_Status st = snapshotFrame(snap, snap_buf);
    if (st != CG_OK) return st;

    // Prefer caller-supplied ROI; fall back to the globally-stored ROI
    int32_t fx = roi_x, fy = roi_y, fw = roi_w, fh = roi_h;
    if (fw <= 0 || fh <= 0)
    {
        lock_guard<mutex> lk(g_roi_mutex);
        if (g_has_roi)
        {
            fx = g_roi_rect.x;     fy = g_roi_rect.y;
            fw = g_roi_rect.width; fh = g_roi_rect.height;
        }
    }

    return inspectFrame(snap, out, fx, fy, fw, fh);
}

CG_API CG_Status CG_InspectICFromFrame(const CG_Frame*     frame,
                                         int32_t             roi_x,
                                         int32_t             roi_y,
                                         int32_t             roi_w,
                                         int32_t             roi_h, 
                                         int32_t             expected_dot_quadrant,
                                         CG_DetectionResult* out)
{
    (void)expected_dot_quadrant;
    if (!frame || !frame->data) return CG_ERR_INVALID_ARG;

    // If caller did not supply a ROI, fall back to the globally-stored one
    int32_t fx = roi_x, fy = roi_y, fw = roi_w, fh = roi_h;
    if (fw <= 0 || fh <= 0)
    {
        lock_guard<mutex> lk(g_roi_mutex);
        if (g_has_roi)
        {
            fx = g_roi_rect.x;     fy = g_roi_rect.y;
            fw = g_roi_rect.width; fh = g_roi_rect.height;
        }
    }

    return inspectFrame(*frame, out, fx, fy, fw, fh);
}

CG_API CG_Status CG_GetLastICResult(CG_ICResult* out)
{
    if (!out) return CG_ERR_INVALID_ARG;
    lock_guard<mutex> lk(g_result_mutex);
    if (!g_has_ic_result)
    {
        setError("CG_GetLastICResult: no inspection has been run yet");
        return CG_ERR_GRAB_FAILED;
    }
    *out = g_last_ic_result;
    return CG_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Crop control  — call before or during streaming; takes effect on the next frame
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
//  FOV control
//  CG_SetFOV  — lock the live stream to a sub-region of the rotated frame.
//               Every frame stored in g_last_frame, delivered to the callback,
//               and used by snapshot/inspection will be this cropped region only.
//               x, y, w, h are in post-rotation pixel coordinates.
//               Pass w=0 or h=0 to disable (same as CG_ClearFOV).
//               Safe to call before or during streaming; takes effect on the
//               very next grabbed frame.
// ─────────────────────────────────────────────────────────────────────────────
CG_API CG_Status CG_SetFOV(int32_t x, int32_t y, int32_t w, int32_t h)
{
    if (w <= 0 || h <= 0)
    {
        lock_guard<mutex> lk(g_crop_mutex);
        g_has_crop  = false;
        g_crop_rect = Rect{};
        printf("[CG_SetFOV] FOV disabled — full frame will be streamed.\n");
        fflush(stdout);
        return CG_OK;
    }
    lock_guard<mutex> lk(g_crop_mutex);
    g_crop_rect = Rect(x, y, w, h);
    g_has_crop  = true;
    printf("[CG_SetFOV] FOV set: x=%d y=%d w=%d h=%d\n", x, y, w, h);
    fflush(stdout);
    return CG_OK;
}

CG_API void CG_ClearFOV(void)
{
    lock_guard<mutex> lk(g_crop_mutex);
    g_has_crop  = false;
    g_crop_rect = Rect{};
    printf("[CG_ClearFOV] FOV cleared — full frame will be streamed.\n");
    fflush(stdout);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────────────────────────────────────
CG_API const char* CG_GetLastError(void)
{
    // Static buffer — valid until the next CG_ call on any thread
    static char buf[512];
    lock_guard<mutex> lk(g_err_mutex);
    strncpy_s(buf, sizeof(buf), g_last_error_str.c_str(), _TRUNCATE);
    return buf;
}

CG_API void CG_GetVersion(int* major, int* minor, int* patch)
{
    if (major) *major = VER_MAJOR;
    if (minor) *minor = VER_MINOR;
    if (patch) *patch = VER_PATCH;
}

CG_API void CG_GetWorkDir(char* buf, int size)
{
    if (!buf || size <= 0) return;
    strncpy_s(buf, size, g_work_dir.c_str(), _TRUNCATE);
}