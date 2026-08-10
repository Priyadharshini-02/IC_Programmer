#include "camera_grabber_api.h"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// =============================================================================
//  Helpers
// =============================================================================

// Fetch the latest frame from the DLL into a cv::Mat.
// Returns an empty Mat if no frame is available yet.
static Mat grabDisplayFrame()
{
    // First call with data == nullptr to query dimensions
    CG_Frame info{};
    if (CG_GetFrame(&info) != CG_OK || info.width == 0 || info.height == 0)
        return {};

    // Allocate and fill
    Mat bgr(info.height, info.width, CV_8UC3);
    CG_Frame fill{};
    fill.data = bgr.data;
    if (CG_GetFrame(&fill) != CG_OK)
        return {};

    return bgr;
}
// Overlay a coloured ROI rectangle and a small status banner on the frame.
static void drawOverlay(Mat& frame,
                        const Rect& roi, bool has_roi,
                        const string& status_line)
{
    if (has_roi)
        rectangle(frame, roi, Scalar(0, 255, 255), 2);

    // Dark banner at the bottom
    int bh = 30;
    Mat banner = frame(Rect(0, frame.rows - bh, frame.cols, bh));
    banner *= 0.4;
    putText(frame, status_line,
            Point(8, frame.rows - 8),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 255), 1);

    // Key legend at the top
    putText(frame, "O:ROI  R:SetRef  G:LoadRef  D:Inspect  V:SaveROI  C:Save25  Q:Quit",
            Point(8, 20),
            FONT_HERSHEY_SIMPLEX, 0.50, Scalar(200, 200, 200), 1);
}

// Continuously save N images of the ROI-cropped region to disk.
// Grabs a fresh live frame on each iteration (not the same frame repeated),
// crops it to the current ROI, and writes it out with a numbered filename.
//
//   out_dir   — destination folder (must already exist)
//   base_name — filename prefix, e.g. "crop" -> crop_001.png, crop_002.png, ...
//   roi       — region to crop from each frame
//   count     — number of images to capture (e.g. 25)
//   delay_ms  — delay between captures in milliseconds
//
// Returns the number of images successfully saved.
static int saveContinuousROICrops(const string& out_dir,
                                   const string& base_name,
                                   const Rect& roi,
                                   int count = 25,
                                   int delay_ms = 100)
{
    if (roi.width <= 0 || roi.height <= 0)
    {
        printf("[ERROR] saveContinuousROICrops: invalid ROI.\n");
        return 0;
    }

    int saved = 0;

    for (int i = 1; i <= count; ++i)
    {
        // Grab a fresh frame for every capture
        Mat frame = grabDisplayFrame();
        if (frame.empty())
        {
            printf("[WARN] [%d/%d] No frame available, skipping.\n", i, count);
            waitKey(delay_ms);
            continue;
        }

        // Clamp ROI to frame bounds to avoid OpenCV throwing on out-of-range crops
        Rect safe_roi = roi & Rect(0, 0, frame.cols, frame.rows);
        if (safe_roi.width <= 0 || safe_roi.height <= 0)
        {
            printf("[WARN] [%d/%d] ROI outside frame bounds, skipping.\n", i, count);
            waitKey(delay_ms);
            continue;
        }

        Mat crop = frame(safe_roi).clone();

        char filename[64];
        snprintf(filename, sizeof(filename), "%s_%03d.png", base_name.c_str(), i);
        string full_path = out_dir + "\\" + filename;

        if (imwrite(full_path, crop))
        {
            printf("[INFO] [%d/%d] Saved: %s\n", i, count, full_path.c_str());
            ++saved;
        }
        else
        {
            printf("[ERROR] [%d/%d] Failed to write: %s\n", i, count, full_path.c_str());
        }

        // Show live progress in the display window
        imshow("IC Inspection", crop);
        waitKey(delay_ms);
    }

    printf("[INFO] Continuous capture complete: %d/%d images saved.\n", saved, count);
    return saved;
}

// Continuously save N full (uncropped) frames to disk.
// Grabs a fresh live frame on each iteration and writes it out in full,
// with no ROI cropping applied.
//
//   out_dir   — destination folder (must already exist)
//   base_name — filename prefix, e.g. "frame" -> frame_001.png, frame_002.png, ...
//   count     — number of images to capture (e.g. 25)
//   delay_ms  — delay between captures in milliseconds
//
// Returns the number of images successfully saved.
static int saveContinuousFrames(const string& out_dir,
                                 const string& base_name,
                                 int count = 25,
                                 int delay_ms = 100)
{
    int saved = 0;

    for (int i = 1; i <= count; ++i)
    {
        // Grab a fresh frame for every capture
        Mat frame = grabDisplayFrame();
        if (frame.empty())
        {
            printf("[WARN] [%d/%d] No frame available, skipping.\n", i, count);
            waitKey(delay_ms);
            continue;
        }

        char filename[64];
        snprintf(filename, sizeof(filename), "%s_%03d.png", base_name.c_str(), i);
        string full_path = out_dir + "\\" + filename;

        if (imwrite(full_path, frame))
        {
            printf("[INFO] [%d/%d] Saved: %s\n", i, count, full_path.c_str());
            ++saved;
        }
        else
        {
            printf("[ERROR] [%d/%d] Failed to write: %s\n", i, count, full_path.c_str());
        }

        // Show live progress in the display window
        imshow("IC Inspection", frame);
        waitKey(delay_ms);
    }

    printf("[INFO] Continuous capture complete: %d/%d images saved.\n", saved, count);
    return saved;
}

// Print a CG_ICResult to stdout in a readable form.
static void printResult(const CG_ICResult& r)
{
    printf("\n========================================\n");
    const char* label = "UNKNOWN";
    switch (r.status)
    {
        case CG_IC_PASS:              label = "PASS";              break;
        case CG_IC_ABSENT:            label = "ABSENT";            break;
        case CG_IC_WRONG_ORIENTATION: label = "WRONG ORIENTATION"; break;
        case CG_IC_PIN_DAMAGED:       label = "PIN DAMAGED";       break;
        case CG_IC_EXTRA:             label = "EXTRA IC";          break;
        case CG_IC_ERROR:             label = "ERROR";             break;
    }
    printf("  Result  : %s\n", label);
    printf("  Message : %s\n", r.message);
    printf("  Present : %d   Absent : %d   Extra : %d\n",
           r.present_count, r.absent_count, r.extra_count);
    printf("  WrongOrient : %d   PinFail : %d\n",
           r.wrong_orientation_count, r.pin_fail_count);
    printf("  Rotation : %.2f deg  step~%d deg\n",
           r.rotation_deg, r.rotation_step);
    printf("  Pins — missing:%d  bent:%d  bridged:%d\n",
           r.pin_missing, r.pin_bent, r.pin_bridged);
    printf("========================================\n\n");
}

// =============================================================================
//  Main
// =============================================================================
int main()
{
    // ── 1. Initialise DLL ────────────────────────────────────────────────────
    if (CG_Initialize() != CG_OK)
    {
        fprintf(stderr, "[ERROR] CG_Initialize failed: %s\n", CG_GetLastError());
        return 1;
    }
    printf("[INFO] DLL initialised.\n");

    // ── 2. Start camera stream ───────────────────────────────────────────────
    if (CG_StartStream() != CG_OK)
    {
        fprintf(stderr, "[ERROR] CG_StartStream failed: %s\n", CG_GetLastError());
        CG_Shutdown();
        return 1;
    }
    printf("[INFO] Camera streaming. Press keys in the display window.\n");

    // ── 3. Application state ─────────────────────────────────────────────────
    Rect   roi;
    bool   has_roi      = false;
    bool   has_ref      = false;
    string status_line  = "No reference set.";

    const string WIN = "IC Inspection";
    namedWindow(WIN, WINDOW_NORMAL);
    resizeWindow(WIN, 1024, 768);

    // ── 4. Main loop ─────────────────────────────────────────────────────────
    while (true)
    {
        // Fetch and display latest frame
        Mat frame = grabDisplayFrame();
        if (!frame.empty())
        {
            Mat display = frame.clone();
            drawOverlay(display, roi, has_roi, status_line);
            imshow(WIN, display);
        }

        int key = waitKey(30) & 0xFF;
        if (key == -1) continue;

        // ── Q : Quit ──────────────────────────────────────────────────────────
        if (key == 'q' || key == 'Q')
        {
            printf("[INFO] Quitting.\n");
            break;
        }

        // ── O : Draw ROI ──────────────────────────────────────────────────────
        else if (key == 'o' || key == 'O')
        {
            Mat frame_for_roi = grabDisplayFrame();
            if (frame_for_roi.empty())
            {
                printf("[WARN] No frame available for ROI selection.\n");
                continue;
            }

            printf("[INFO] Draw ROI — drag a rectangle, press ENTER/SPACE to confirm.\n");

            // cv::selectROI pauses the loop until the user confirms
            Rect selected = selectROI(WIN, frame_for_roi, false, false);

            if (selected.width > 0 && selected.height > 0)
            {
                roi     = selected;
                has_roi = true;

                CG_Status st = CG_SetROI(roi.x, roi.y, roi.width, roi.height);
                if (st == CG_OK)
                {
                    printf("[ROI] Set: x=%d y=%d w=%d h=%d\n",
                           roi.x, roi.y, roi.width, roi.height);
                    status_line = "ROI set. Press R to set reference.";

                    // ROI change invalidates any existing reference
                    has_ref = false;
                }
                else
                {
                    printf("[ERROR] CG_SetROI failed: %s\n", CG_GetLastError());
                }
            }
            else
            {
                printf("[INFO] ROI selection cancelled.\n");
            }
        }

        // ── R : Set reference from live frame ─────────────────────────────────
        else if (key == 'r' || key == 'R')
        {
            if (!has_roi)
            {
                printf("[WARN] Set a ROI first (press O).\n");
                status_line = "Set ROI first (O), then R.";
                continue;
            }

            printf("[INFO] Setting reference from live frame...\n");
            status_line = "Setting reference...";

            CG_Status st = CG_SetReference();
            if (st == CG_OK)
            {
                has_ref     = true;
                status_line = "Reference SET. Press D to inspect.";
                printf("[INFO] Reference set successfully.\n");
            }
            else
            {
                status_line = string("SetRef FAILED: ") + CG_GetLastError();
                printf("[ERROR] CG_SetReference failed: %s\n", CG_GetLastError());
            }
        }

        // ── G : Load reference from file ──────────────────────────────────────
        else if (key == 'g' || key == 'G')
        {
            // Ask the user to type a file path in the console
            printf("[INFO] Enter path to reference image: ");
            fflush(stdout);

            char path[512] = {};
            if (!fgets(path, sizeof(path), stdin)) continue;

            // Strip trailing newline
            size_t len = strlen(path);
            if (len > 0 && path[len - 1] == '\n') path[len - 1] = '\0';

            int32_t rx = 0, ry = 0, rw = 0, rh = 0;
            if (has_roi)
            {
                rx = roi.x; ry = roi.y;
                rw = roi.width; rh = roi.height;
            }

            CG_Status st = CG_LoadReferenceFromFile(path, rx, ry, rw, rh);
            if (st == CG_OK)
            {
                has_ref     = true;
                status_line = string("Ref loaded: ") + path;
                printf("[INFO] Reference loaded from: %s\n", path);
            }
            else
            {
                status_line = string("LoadRef FAILED: ") + CG_GetLastError();
                printf("[ERROR] CG_LoadReferenceFromFile failed: %s\n",
                       CG_GetLastError());
            }
        }

        // ── D : Inspect current frame ─────────────────────────────────────────
        else if (key == 'd' || key == 'D')
        {
            if (!has_ref)
            {
                printf("[WARN] No reference set. Press R or G first.\n");
                status_line = "No reference — press R or G first.";
                continue;
            }

            // Grab current frame from DLL
            Mat test_frame = grabDisplayFrame();
            if (test_frame.empty())
            {
                printf("[WARN] No frame available for inspection.\n");
                continue;
            }

            printf("[INFO] Inspecting frame...\n");
            status_line = "Inspecting...";

            // Wrap Mat data in CG_Frame — no copy, DLL reads our buffer
            CG_Frame cgf{};
            cgf.data   = test_frame.data;
            cgf.width  = test_frame.cols;
            cgf.height = test_frame.rows;
            cgf.step   = static_cast<int>(test_frame.step);

            int32_t rx = has_roi ? roi.x      : 0;
            int32_t ry = has_roi ? roi.y      : 0;
            int32_t rw = has_roi ? roi.width  : 0;
            int32_t rh = has_roi ? roi.height : 0;

            CG_DetectionResult det{};
            CG_Status st = CG_InspectICFromFrame(&cgf,
                                                  rx, ry, rw, rh,
                                                  0,    // expected_dot_quadrant
                                                  &det);
            if (st != CG_OK)
            {
                status_line = string("Inspect FAILED: ") + CG_GetLastError();
                printf("[ERROR] CG_InspectICFromFrame failed: %s\n",
                       CG_GetLastError());
                continue;
            }

            // Retrieve and print the rich result
            CG_ICResult rich{};
            if (CG_GetLastICResult(&rich) == CG_OK)
                printResult(rich);

            // Update status banner based on outcome
            switch (rich.status)
            {
                case CG_IC_PASS:
                    status_line = "PASS";
                    break;
                case CG_IC_ABSENT:
                    status_line = "FAIL — IC ABSENT";
                    break;
                case CG_IC_WRONG_ORIENTATION:
                    status_line = string("FAIL — WRONG ORIENTATION  ")
                                + to_string((int)rich.rotation_deg) + " deg";
                    break;
                case CG_IC_PIN_DAMAGED:
                    status_line = "FAIL — PIN DAMAGE";
                    break;
                case CG_IC_EXTRA:
                    status_line = "FAIL — EXTRA IC";
                    break;
                default:
                    status_line = "ERROR";
                    break;
            }
        }
        else if (key == 's' || key == 'S')
        {
            // Ask the user to type a destination path in the console
            printf("[INFO] Enter output path for saved frame (e.g. frame.png): ");
            fflush(stdout);

            char save_path[512] = {};
            if (!fgets(save_path, sizeof(save_path), stdin)) continue;

            // Strip trailing newline / carriage return
            size_t slen = strlen(save_path);
            while (slen > 0 && (save_path[slen-1] == '\n' || save_path[slen-1] == '\r'))
                save_path[--slen] = '\0';

            CG_Status st = CG_SaveFrameToPath(save_path);
            if (st == CG_OK)
            {
                status_line = string("Frame saved: ") + save_path;
                printf("[INFO] Frame saved to: %s\n", save_path);
            }
            else
            {
                status_line = string("SaveFrame FAILED: ") + CG_GetLastError();
                printf("[ERROR] CG_SaveFrameToPath failed: %s\n", CG_GetLastError());
            }
        }

        // ── V : Save ROI region of current frame to disk ─────────────────────
        else if (key == 'v' || key == 'V')
        {
            if (!has_roi)
            {
                printf("[WARN] No ROI set. Press O to draw a ROI first.\n");
                status_line = "Set ROI first (O), then V.";
                continue;
            }

            // Ask the user to type a destination path in the console
            printf("[INFO] Enter output path for ROI frame (e.g. C:\\frames\\frame_name.png): ");
            fflush(stdout);

            char roi_path[512] = {};
            if (!fgets(roi_path, sizeof(roi_path), stdin)) continue;

            // Strip trailing newline / carriage return
            size_t rlen = strlen(roi_path);
            while (rlen > 0 && (roi_path[rlen-1] == '\n' || roi_path[rlen-1] == '\r'))
                roi_path[--rlen] = '\0';

            CG_Status st = CG_SaveFrame_roi(roi_path,
                                             roi.x, roi.y,
                                             roi.width, roi.height);
            if (st == CG_OK)
            {
                status_line = string("ROI saved: ") + roi_path;
                printf("[INFO] ROI frame saved to: %s\n", roi_path);
            }
            else
            {
                status_line = string("SaveFrame_roi FAILED: ") + CG_GetLastError();
                printf("[ERROR] CG_SaveFrame_roi failed: %s\n", CG_GetLastError());
            }
        }

        // ── C : Continuously save 25 ROI crops to disk ───────────────────────
        else if (key == 'c' || key == 'C')
        {
        
    printf("[INFO] Enter output folder for 25 full-frame captures (e.g. C:\\frames): ");
    fflush(stdout);

    char dir_path[512] = {};
    if (!fgets(dir_path, sizeof(dir_path), stdin)) continue;

    // Strip trailing newline / carriage return
    size_t dlen = strlen(dir_path);
    while (dlen > 0 && (dir_path[dlen-1] == '\n' || dir_path[dlen-1] == '\r'))
        dir_path[--dlen] = '\0';

    printf("[INFO] Capturing 25 full frames...\n");
    status_line = "Capturing 25 frames...";

    // int saved = saveContinuousFrames(dir_path, "frame", 25, 100);
    int saved = saveContinuousROICrops(dir_path, "frame", roi, 25, 100);

    status_line = "Saved " + to_string(saved) + "/25 frames to " + dir_path;
    printf("[INFO] %s\n", status_line.c_str());
}
        

    }

    // ── 5. Cleanup ───────────────────────────────────────────────────────────
    destroyAllWindows();
    CG_StopStream();
    CG_Shutdown();
    printf("[INFO] Done.\n");
    return 0;
}