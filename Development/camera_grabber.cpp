/*  camera_grabber.cpp  (v4 — thin UI shell)
 *
 * All IC-inspection pipeline logic has moved into camera_grabber_lib.dll (v4).
 * This file is now a pure UI shell: it handles the OpenCV window, mouse-drawn
 * ROI, key bindings, and result display.  It makes NO direct Python calls and
 * contains NO file-system helpers — everything is delegated to the DLL.
 *
 * Key bindings
 * ────────────
 *   o     — enter ROI draw mode  (click-drag to define bounding box)
 *   ENTER — confirm drawn ROI    (calls CG_SetROI → DLL stores it)
 *   ESC   — cancel ROI draw mode
 *   r     — set reference        (DLL: YOLO check → crop → save ref)
 *   d     — run IC inspection    (DLL: full Python pipeline in bg thread)
 *   q     — quit
 *
 * What was removed vs the v3 camera_grabber.cpp
 * ──────────────────────────────────────────────
 *   • checkICPresenceInROI()  — now inside DLL (CG_SetReference)
 *   • setReference()          — now inside DLL (CG_SetReference /
 *                               CG_SetReferenceFromFrame)
 *   • runFullInspection()     — now inside DLL (CG_InspectICFromFrame)
 *   • parseAfterInt/Float()   — now inside DLL
 *   • skipTokenAndSpaces()    — now inside DLL
 *   • fileExists()            — now inside DLL
 *   • pathJoin() / P()        — now inside DLL
 *   • current_ref_path        — no longer needed (DLL owns ref paths)
 *   • ref_snap / pre-crop     — no longer needed
 *   • CG_Get_Version typo     — fixed to CG_GetVersion
 *   • #include <fstream>      — removed (no direct file I/O here)
 *   • #include <sstream>      — removed (no string building here)
 *   • directory creation in main — removed (DLL does it in CG_Initialize)
 *
 * Inspection flow (v4)
 * ────────────────────
 *   'd' key pressed
 *     │
 *     ├─ snapshot frame into snap_buf
 *     ├─ detach background thread:
 *     │     CG_InspectICFromFrame(&cgf_snap, roi.x, roi.y, roi.w, roi.h, 0, &dr)
 *     │       └─ DLL: crop → save current_test.png → Python → parse output
 *     │     CG_GetLastICResult(&det)   ← rich CG_ICResult for printICResult()
 *     │     load annotated output image
 *     └─ main thread: display result overlay + HUD
 */

#include <cstdint>
#ifndef _UINT8_T_DEFINED
  #define _UINT8_T_DEFINED
#endif
#ifndef _UINT32_T_DEFINED
  #define _UINT32_T_DEFINED
#endif
#ifndef _INT32_T_DEFINED
  #define _INT32_T_DEFINED
#endif
#ifndef _INT64_T_DEFINED
  #define _INT64_T_DEFINED
#endif

#define NOMINMAX
#include <windows.h>

#include <opencv2/opencv.hpp>
#include "camera_grabber_api.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <signal.h>

using namespace std;
using namespace cv;

// ─────────────────────────────────────────────────────────────────────────────
//  Signal handler
// ─────────────────────────────────────────────────────────────────────────────
static atomic<bool> keep_running{true};
void signal_handler(int) { keep_running.store(false); }

// ─────────────────────────────────────────────────────────────────────────────
//  ROI UI state  (display-side only — no file I/O)
// ─────────────────────────────────────────────────────────────────────────────
struct ROIState
{
    bool  active  = false;  ///< Currently in draw mode
    bool  drawing = false;  ///< Mouse button held down
    bool  defined = false;  ///< A valid ROI has been confirmed
    Point start, end;
    Rect  roi_display;      ///< Rectangle in display-pixel space
    Rect  roi_orig;         ///< Rectangle in original-frame-pixel space
    float scale_x = 1.f;
    float scale_y = 1.f;
    Mat   frozen;           ///< Frame snapshot shown while drawing
};

static ROIState g_roi;
static mutex    g_roi_mutex;  ///< Protects roi_display / drawing (mouse thread)

// Convert display-space rectangle → original image coordinates
static Rect displayToOrig(const ROIState& rs)
{
    int x1 = static_cast<int>(min(rs.start.x, rs.end.x) / rs.scale_x);
    int y1 = static_cast<int>(min(rs.start.y, rs.end.y) / rs.scale_y);
    int x2 = static_cast<int>(max(rs.start.x, rs.end.x) / rs.scale_x);
    int y2 = static_cast<int>(max(rs.start.y, rs.end.y) / rs.scale_y);
    return Rect(Point(x1, y1), Point(x2, y2));
}

void mouseCallback(int event, int x, int y, int /*flags*/, void* /*user*/)
{
    if (!g_roi.active) return;
    lock_guard<mutex> lk(g_roi_mutex);
    if (event == EVENT_LBUTTONDOWN)
    {
        g_roi.drawing = true;
        g_roi.start = g_roi.end = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE && g_roi.drawing)
    {
        g_roi.end = Point(x, y);
    }
    else if (event == EVENT_LBUTTONUP && g_roi.drawing)
    {
        g_roi.drawing = false;
        g_roi.end = Point(x, y);
        if (abs(g_roi.end.x - g_roi.start.x) > 10 &&
            abs(g_roi.end.y - g_roi.start.y) > 10)
        {
            g_roi.roi_display = Rect(
                Point(min(g_roi.start.x, g_roi.end.x),
                      min(g_roi.start.y, g_roi.end.y)),
                Point(max(g_roi.start.x, g_roi.end.x),
                      max(g_roi.start.y, g_roi.end.y)));
        }
    }
}

void drawROIOverlay(Mat& display, const ROIState& rs)
{
    if (!rs.active) return;

    // Semi-transparent instruction banner
    Mat banner_roi = display(Rect(0, 0, display.cols, 80));
    Mat dark = Mat::zeros(banner_roi.size(), banner_roi.type());
    addWeighted(banner_roi, 0.45, dark, 0.55, 0, banner_roi);

    Scalar col(0, 200, 255);
    putText(display,
            rs.roi_display.area() > 0
                ? "ROI MODE \xe2\x80\x94 Press ENTER to confirm    ESC = cancel"
                : "ROI MODE \xe2\x80\x94 Draw ROI: click and drag",
            Point(15, 30), FONT_HERSHEY_SIMPLEX, 0.58, col, 1);
    putText(display, "ROI is used for reference capture & IC inspection",
            Point(15, 58), FONT_HERSHEY_SIMPLEX, 0.44, Scalar(200, 200, 200), 1);

    Rect r;
    {
        lock_guard<mutex> lk(g_roi_mutex);
        if (rs.drawing)
            r = Rect(Point(min(rs.start.x, rs.end.x), min(rs.start.y, rs.end.y)),
                     Point(max(rs.start.x, rs.end.x), max(rs.start.y, rs.end.y)));
        else
            r = rs.roi_display;
    }
    if (r.area() > 0)
    {
        rectangle(display, r, col, 2);
        putText(display,
                to_string(static_cast<int>(r.width  / rs.scale_x)) + "x" +
                to_string(static_cast<int>(r.height / rs.scale_y)) + " px",
                Point(r.x + 4, r.y + r.height - 6), FONT_HERSHEY_SIMPLEX, 0.42, col, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: wrap a CG_Frame around a Mat (zero-copy, Mat keeps ownership)
// ─────────────────────────────────────────────────────────────────────────────
static Mat cgToMat(const CG_Frame& f)
{
    return Mat(f.height, f.width, CV_8UC3, f.data, static_cast<size_t>(f.step));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Print CG_ICResult to console  (unchanged from v3)
// ─────────────────────────────────────────────────────────────────────────────
static void printICResult(const CG_ICResult& r)
{
    const char* verdict = "UNKNOWN";
    switch (r.status)
    {
        case CG_IC_PASS:              verdict = "PASS";                         break;
        case CG_IC_ABSENT:            verdict = "FAIL \xe2\x80\x94 IC ABSENT";            break;
        case CG_IC_WRONG_ORIENTATION: verdict = "FAIL \xe2\x80\x94 WRONG ORIENTATION";    break;
        case CG_IC_PIN_DAMAGED:       verdict = "FAIL \xe2\x80\x94 PIN DAMAGED";          break;
        case CG_IC_EXTRA:             verdict = "FAIL \xe2\x80\x94 EXTRA IC";             break;
        case CG_IC_ERROR:             verdict = "ERROR";                        break;
    }

    cout << "\n"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
            "\xe2\x95\x90\n"
         << "  RESULT: " << verdict << "\n"
         << "  Present (matched)  : " << r.present_count           << "\n"
         << "  Absent  (missing)  : " << r.absent_count            << "\n"
         << "  Extra (unexpected) : " << r.extra_count             << "\n"
         << "  Wrong orientation  : " << r.wrong_orientation_count << "\n";

    if (r.wrong_orientation_count > 0)
        cout << "    rotation         : " << r.rotation_deg  << " deg\n"
             << "    nearest 90\xc2\xb0 step : " << r.rotation_step << " deg\n";

    cout << "  Pin failures       : " << r.pin_fail_count << "\n";
    if (r.pin_fail_count > 0)
        cout << "    missing          : " << r.pin_missing << "\n"
             << "    bent             : " << r.pin_bent    << "\n"
             << "    bridged          : " << r.pin_bridged << "\n";

    cout << "  Message            : " << r.message << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Load the annotated result image written by Python (if present)
// ─────────────────────────────────────────────────────────────────────────────
static Mat loadLatestAnnotated()
{
    // Path is always relative to the DLL's parent directory, but imread works
    // fine with relative paths if the CWD is the parent dir.  The DLL manages
    // the canonical path; we only need the readable suffix here.
    char work[1024] = {};
    CG_GetWorkDir(work, sizeof(work));
    string annotated = string(work) + "\\..\\output\\test_annotated.png";
    Mat img = imread(annotated);
    return img;  // empty Mat if not found — caller checks .empty()
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int /*argc*/, char* /*argv*/[])
{
    signal(SIGINT, signal_handler);

    // ── Init DLL ─────────────────────────────────────────────────────────────
    if (CG_Initialize() != CG_OK)
    {
        cerr << "CG_Initialize failed: " << CG_GetLastError() << endl;
        return 1;
    }
    {
        int ma, mi, pa;
        CG_GetVersion(&ma, &mi, &pa);   // fixed: was CG_Get_Version
        cout << "camera_grabber_lib v" << ma << "." << mi << "." << pa << "\n";
    }
    // NOTE: directories (ref\, images\, tmp\, output\) are created inside
    //       CG_Initialize() — no need to call createDirectory() here.

    int exitCode = 0;
    try
    {
        // ── Start stream ─────────────────────────────────────────────────────
        if (CG_StartStream() != CG_OK)
        {
            cerr << "CG_StartStream failed: " << CG_GetLastError() << endl;
            CG_Shutdown(); return 1;
        }
        cout << "Stream started.\n";

        // ── Wait for first valid frame to get resolution ──────────────────
        int fw = 0, fh = 0;
        for (int i = 0; i < 200 && keep_running.load(); ++i)
        {
            CG_Frame p{};
            if (CG_GetFrame(&p) == CG_OK && p.width > 0)
            {
                fw = p.width; fh = p.height; break;
            }
            this_thread::sleep_for(chrono::milliseconds(50));
        }
        if (!fw)
        {
            cerr << "No frame received." << endl;
            CG_StopStream(); CG_Shutdown(); return 1;
        }
        cout << "Resolution: " << fw << " x " << fh << "\n";

        // ── Per-frame pixel buffer (reused every loop iteration) ──────────
        vector<uint8_t> frame_buf(static_cast<size_t>(fw) * fh * 3);

        // ── UI / inspection state ─────────────────────────────────────────
        bool         has_reference = false;
        atomic<bool> processing{false};
        Mat          result_overlay;
        CG_ICResult  last_result   = {};
        bool         last_valid    = false;
        mutex        result_mutex;

        // ── OpenCV window ─────────────────────────────────────────────────
        const string WIN    = "IC Inspection - Live Stream";
        const int    DISP_W = 800, DISP_H = 600;
        namedWindow(WIN, WINDOW_NORMAL);
        setMouseCallback(WIN, mouseCallback, nullptr);

        // ── Main display loop ─────────────────────────────────────────────
        while (keep_running.load())
        {
            // Grab the latest frame into our buffer
            CG_Frame cgf{};
            cgf.data   = frame_buf.data();
            cgf.width  = fw;
            cgf.height = fh;
            cgf.step   = fw * 3;

            if (CG_GetFrame(&cgf) != CG_OK)
            {
                this_thread::sleep_for(chrono::milliseconds(5));
                continue;
            }

            // Handle resolution change
            if (cgf.width != fw || cgf.height != fh)
            {
                fw = cgf.width; fh = cgf.height;
                frame_buf.assign(static_cast<size_t>(fw) * fh * 3, 0);
                continue;
            }

            // Build the display Mat
            Mat frame   = cgToMat(cgf).clone();
            Mat display = g_roi.active ? g_roi.frozen.clone() : frame.clone();

            // Overlay annotated inspection result (when not in ROI draw mode)
            if (!result_overlay.empty() && !g_roi.active)
            {
                Mat ov = result_overlay.clone();
                if (ov.size() != display.size()) resize(ov, ov, display.size());
                display = ov;
            }

            // Update display scale factors for mouse ↔ original coordinate mapping
            int src_c = g_roi.active ? g_roi.frozen.cols : display.cols;
            int src_r = g_roi.active ? g_roi.frozen.rows : display.rows;
            g_roi.scale_x = static_cast<float>(DISP_W) / src_c;
            g_roi.scale_y = static_cast<float>(DISP_H) / src_r;

            Mat resized;
            resize(display, resized, Size(DISP_W, DISP_H));

            // Draw confirmed ROI box on live view
            if (g_roi.defined && !g_roi.active)
            {
                int dx1 = static_cast<int>(g_roi.roi_orig.x                           * g_roi.scale_x);
                int dy1 = static_cast<int>(g_roi.roi_orig.y                           * g_roi.scale_y);
                int dx2 = static_cast<int>((g_roi.roi_orig.x + g_roi.roi_orig.width)  * g_roi.scale_x);
                int dy2 = static_cast<int>((g_roi.roi_orig.y + g_roi.roi_orig.height) * g_roi.scale_y);
                rectangle(resized, Point(dx1, dy1), Point(dx2, dy2), Scalar(0, 255, 0), 2);
                putText(resized, "ROI", Point(dx1 + 4, dy1 + 18),
                        FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 1);
            }

            drawROIOverlay(resized, g_roi);

            // ── HUD ───────────────────────────────────────────────────────
            {
                string hud; Scalar hud_col(180, 180, 180);
                if (processing.load())
                {
                    hud = "Inspecting..."; hud_col = Scalar(200, 200, 0);
                }
                else if (last_valid)
                {
                    bool pass = (last_result.status == CG_IC_PASS);
                    hud     = pass ? "RESULT: PASS" : "RESULT: FAIL";
                    hud_col = pass ? Scalar(0, 220, 0) : Scalar(0, 0, 220);
                    if (!pass)
                    {
                        switch (last_result.status)
                        {
                            case CG_IC_ABSENT:            hud += " (ABSENT)";       break;
                            case CG_IC_WRONG_ORIENTATION: hud += " (WRONG ORIENT)"; break;
                            case CG_IC_PIN_DAMAGED:       hud += " (PIN DAMAGED)";  break;
                            case CG_IC_EXTRA:             hud += " (EXTRA IC)";     break;
                            case CG_IC_ERROR:             hud += " (ERROR)";        break;
                            default: break;
                        }
                    }
                    hud += string("  |  ") + (has_reference  ? "Ref OK"  : "No Ref");
                    hud += string("  |  ") + (g_roi.defined  ? "ROI OK"  : "No ROI");
                }
                else
                {
                    hud  = has_reference ? "Ref OK"  : "No Ref";
                    hud += string("  |  ") + (g_roi.defined  ? "ROI OK"  : "No ROI");
                }
                putText(resized, "o=ROI  r=Reference  d=Inspect  q=Quit",
                        Point(10, DISP_H - 30), FONT_HERSHEY_SIMPLEX, 0.42, Scalar(140, 140, 140), 1);
                putText(resized, hud,
                        Point(10, DISP_H - 10), FONT_HERSHEY_SIMPLEX, 0.50, hud_col, 1);
            }

            imshow(WIN, resized);
            int key = waitKey(1);

            // ── 'o': enter ROI draw mode ──────────────────────────────────
            if ((key == 'o' || key == 'O') && !processing.load() && !g_roi.active)
            {
                g_roi.active      = true;
                g_roi.drawing     = false;
                g_roi.roi_display = Rect();
                g_roi.frozen      = frame.clone();
                cout << "[ROI] Draw mode ON.\n";
            }

            // ── ENTER: confirm drawn ROI ──────────────────────────────────
            else if (key == 13 && g_roi.active)
            {
                if (g_roi.roi_display.area() > 0)
                {
                    g_roi.roi_orig  = displayToOrig(g_roi);
                    g_roi.roi_orig &= Rect(0, 0, fw, fh);

                    if (g_roi.roi_orig.area() > 0)
                    {
                        g_roi.defined = true;
                        g_roi.active  = false;

                        // ── Tell the DLL about the new ROI ────────────────
                        CG_Status st = CG_SetROI(
                            g_roi.roi_orig.x,     g_roi.roi_orig.y,
                            g_roi.roi_orig.width, g_roi.roi_orig.height);
                        if (st != CG_OK)
                            cerr << "[ROI] CG_SetROI warning: " << CG_GetLastError() << "\n";

                        // Changing ROI invalidates any previous reference
                        has_reference = false;
                        result_overlay = Mat();
                        last_result = {}; last_valid = false;

                        cout << "[ROI] Confirmed: x=" << g_roi.roi_orig.x
                             << " y=" << g_roi.roi_orig.y
                             << " w=" << g_roi.roi_orig.width
                             << " h=" << g_roi.roi_orig.height << "\n";
                    }
                    else cout << "[ROI] Outside image bounds.\n";
                }
                else cout << "[ROI] No box drawn yet.\n";
            }

            // ── ESC: cancel ROI draw ──────────────────────────────────────
            else if (key == 27 && g_roi.active)
            {
                g_roi.active  = false;
                g_roi.drawing = false;
                cout << "[ROI] Cancelled.\n";
            }

            // ── 'r': set reference via DLL ────────────────────────────────
            //   CG_SetReference() grabs the latest live frame, crops it to
            //   the stored ROI, runs YOLO to confirm IC presence, and saves
            //   the crop as ref\current_reference.png — all inside the DLL.
            else if ((key == 'r' || key == 'R') && !g_roi.active && !processing.load())
            {
                cout << "[REF] Checking IC presence and saving reference...\n";

                result_overlay = Mat(); last_result = {}; last_valid = false;

                CG_Status st = CG_SetReference();
                if (st == CG_OK)
                {
                    has_reference = true;
                    cout << "[REF] Reference set successfully.\n";
                }
                else if (st == CG_ERR_NO_BOARD)
                {
                    cout << "[REF] REJECTED \xe2\x80\x94 No IC detected in ROI."
                            " Place IC and retry.\n";
                }
                else
                {
                    cerr << "[REF] Failed: " << CG_GetLastError() << "\n";
                }
            }

            // ── 'd': full IC inspection in background thread ───────────────
            //   Snapshot the current frame, then hand it to the DLL via
            //   CG_InspectICFromFrame().  The DLL crops, saves the test
            //   image, runs Python, and parses the output.  We retrieve the
            //   rich CG_ICResult afterwards with CG_GetLastICResult().
            else if ((key == 'd' || key == 'D') && !g_roi.active && !processing.load())
            {
                if (!has_reference)
                {
                    cout << "[INSPECT] Set reference first (press 'r').\n";
                    continue;
                }

                processing.store(true);
                result_overlay = Mat(); last_result = {}; last_valid = false;
                cout << "\n=== RUNNING IC INSPECTION ===\n";

                // Snapshot the frame NOW so the background thread is fully
                // decoupled from the live camera buffer.
                vector<uint8_t> snap_buf(frame_buf);
                CG_Frame cgf_snap  = cgf;
                cgf_snap.data      = snap_buf.data();
                Rect roi_snap      = g_roi.roi_orig;   // copy for lambda capture

                thread([&, cgf_snap, snap_buf = std::move(snap_buf), roi_snap]() mutable
                {
                    cgf_snap.data = snap_buf.data();

                    // The DLL does: crop → save current_test.png → Python → parse
                    CG_DetectionResult dr{};
                    CG_Status st = CG_InspectICFromFrame(
                        &cgf_snap,
                        roi_snap.x,     roi_snap.y,
                        roi_snap.width, roi_snap.height,
                        /*expected_dot_quadrant=*/ 0,
                        &dr);

                    CG_ICResult det{};
                    if (st == CG_OK)
                    {
                        // Retrieve the rich result (pin counts, rotation, etc.)
                        CG_GetLastICResult(&det);
                        printICResult(det);
                    }
                    else
                    {
                        cerr << "[INSPECT] CG_InspectICFromFrame error: "
                             << CG_GetLastError() << "\n";
                        det.status = CG_IC_ERROR;
                        snprintf(det.message, sizeof(det.message),
                                 "DLL error: %s", CG_GetLastError());
                    }

                    Mat ann = loadLatestAnnotated();

                    {
                        lock_guard<mutex> lk(result_mutex);
                        result_overlay = ann;
                        last_result    = det;
                        last_valid     = true;
                    }
                    processing.store(false);

                }).detach();
            }

            // ── 'q': quit ─────────────────────────────────────────────────
            else if (key == 'q' || key == 'Q')
            {
                keep_running.store(false);
                break;
            }

        }  // end while(keep_running)

        destroyAllWindows();
        CG_StopStream();
    }
    catch (const exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        exitCode = 1;
    }

    CG_Shutdown();
    cout << "\n.......stream stopped....!\n";
    return exitCode;
}