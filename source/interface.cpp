//
// Created by 10633 on 2025/6/12.
//
#include "interface.h"

#include <fstream>
#include <iostream>


ScreenCapture::ScreenCapture() : overlayWnd(nullptr), isSelecting(false) {
    startPoint = {0, 0};
    endPoint = {0, 0};
    selectedRect = {0, 0, 0, 0};
}

ScreenCapture::~ScreenCapture() {
    if (overlayWnd) {
        DestroyWindow(overlayWnd);
        overlayWnd = nullptr;
    }
}

LRESULT CALLBACK ScreenCapture::OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    ScreenCapture *pThis = reinterpret_cast<ScreenCapture *>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg) {
        case WM_LBUTTONDOWN:
            if (pThis) {
                pThis->isSelecting = true;
                pThis->startPoint.x = LOWORD(lParam);
                pThis->startPoint.y = HIWORD(lParam);
                pThis->endPoint = pThis->startPoint;
                SetCapture(hwnd);
            }
            break;

        case WM_MOUSEMOVE:
            if (pThis && pThis->isSelecting) {
                pThis->endPoint.x = LOWORD(lParam);
                pThis->endPoint.y = HIWORD(lParam);
                InvalidateRect(hwnd, NULL, TRUE);
            }
            break;

        case WM_LBUTTONUP:
            if (pThis && pThis->isSelecting) {
                pThis->isSelecting = false;
                pThis->endPoint.x = LOWORD(lParam);
                pThis->endPoint.y = HIWORD(lParam);

                pThis->selectedRect.left = std::min(pThis->startPoint.x, pThis->endPoint.x);
                pThis->selectedRect.top = std::min(pThis->startPoint.y, pThis->endPoint.y);
                pThis->selectedRect.right = std::max(pThis->startPoint.x, pThis->endPoint.x);
                pThis->selectedRect.bottom = std::max(pThis->startPoint.y, pThis->endPoint.y);

                ReleaseCapture();
                PostMessage(hwnd, WM_CLOSE, 0, 0);
            }
            break;

        case WM_PAINT:
            if (pThis) {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);

                if (pThis->isSelecting) {
                    HPEN pen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));
                    HPEN oldPen = (HPEN) SelectObject(hdc, pen);

                    Rectangle(hdc,
                              std::min(pThis->startPoint.x, pThis->endPoint.x),
                              std::min(pThis->startPoint.y, pThis->endPoint.y),
                              std::max(pThis->startPoint.x, pThis->endPoint.x),
                              std::max(pThis->startPoint.y, pThis->endPoint.y));

                    SelectObject(hdc, oldPen);
                    DeleteObject(pen);
                }

                EndPaint(hwnd, &ps);
            }
            break;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostMessage(hwnd, WM_CLOSE, 0, 0);
            }
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

bool ScreenCapture::CreateOverlayWindow() {
    const wchar_t *className = L"ScreenCaptureOverlay";

    WNDCLASSW wc = {};
    wc.lpfnWndProc = OverlayWndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(NULL, IDC_CROSS);
    wc.hbrBackground = (HBRUSH) GetStockObject(NULL_BRUSH);

    if (!RegisterClassW(&wc)) {
        return false;
    }

    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    overlayWnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        className,
        L"Screen Capture Overlay",
        WS_POPUP,
        0, 0, screenWidth, screenHeight,
        NULL, NULL, GetModuleHandle(NULL), NULL
    );

    if (!overlayWnd) {
        return false;
    }

    SetLayeredWindowAttributes(overlayWnd, 0, 50, LWA_ALPHA);
    SetWindowLongPtr(overlayWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
    ShowWindow(overlayWnd, SW_SHOW);
    UpdateWindow(overlayWnd);

    return true;
}

cv::Mat ScreenCapture::CaptureScreenRegion(const RECT &rect) {
    int width = rect.right - rect.left;
    int height = rect.bottom - rect.top;

    if (width <= 0 || height <= 0) {
        return cv::Mat();
    }

    HDC screenDC = GetDC(NULL);
    HDC memDC = CreateCompatibleDC(screenDC);

    BITMAPINFOHEADER bmi = {0};
    bmi.biSize = sizeof(BITMAPINFOHEADER);
    bmi.biWidth = width;
    bmi.biHeight = -height;
    bmi.biPlanes = 1;
    bmi.biBitCount = 32;
    bmi.biCompression = BI_RGB;

    void *pBits = nullptr;
    HBITMAP bitmap = CreateDIBSection(screenDC, (BITMAPINFO *) &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!bitmap) {
        DeleteDC(memDC);
        ReleaseDC(NULL, screenDC);
        return cv::Mat();
    }

    HBITMAP oldBitmap = (HBITMAP) SelectObject(memDC, bitmap);
    BitBlt(memDC, 0, 0, width, height, screenDC, rect.left, rect.top, SRCCOPY);

    // 创建Mat并复制数据(Win32默认是RGB格式，但opencv默认是BGR)
    cv::Mat mat(height, width, CV_8UC3);
    BYTE *src = static_cast<BYTE *>(pBits);
    int srcStride = width * 4;

    for (int y = 0; y < height; y++) {
        BYTE *srcRow = src + y * srcStride;
        BYTE *dstRow = mat.ptr<BYTE>(y);
        for (int x = 0; x < width; x++) {
            dstRow[0] = srcRow[0];
            dstRow[1] = srcRow[1];
            dstRow[2] = srcRow[2];
            srcRow += 4;
            dstRow += 3;
        }
    }

    SelectObject(memDC, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(memDC);
    ReleaseDC(NULL, screenDC);

    mat.convertTo(mat,CV_8UC1);
    return mat;
}

bool ScreenCapture::StartCapture() {
    if (!CreateOverlayWindow()) {
        return false;
    }

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return true;
}

RECT ScreenCapture::GetSelectedRect() const {
    return selectedRect;
}


int show_result(const std::vector<rec_result> &texts) {
    std::ofstream out("ocr_results.txt", std::ios::out | std::ios::binary);
    out << "\xEF\xBB\xBF";

    for (const auto &text: texts) {
        out << text.text << std::endl;
    }

    auto it = std::min_element(
        texts.begin(), texts.end(),
        [](const rec_result &a, const rec_result &b) {
            if (a.score == 0) return false;
            if (b.score == 0) return true;
            return a.score < b.score;
        }
    );
    int targetLine = distance(texts.begin(), it);

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    PathRemoveFileSpecW(exePath);

    wchar_t filePath[MAX_PATH];
    PathCombineW(filePath, exePath, L"ocr_results.txt");

    std::wstring commandLine = L"notepad.exe" L" \"" + std::wstring(filePath) + L"\"";

    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;

    if (!CreateProcessW(
        NULL,
        &commandLine[0],
        NULL, NULL, FALSE,
        0, NULL, NULL,
        &si, &pi)) {
        std::wcerr << L"Error：" << GetLastError() << std::endl;
        return 1;
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}
