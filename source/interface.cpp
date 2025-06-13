//
// Created by 10633 on 2025/6/12.
//
// ScreenCapture.cpp
#include "interface.h"
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

int ScreenCapture::GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0;
    UINT size = 0;
    ImageCodecInfo* pImageCodecInfo = NULL;

    GetImageEncodersSize(&num, &size);
    if (size == 0) return -1;

    pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
    if (pImageCodecInfo == NULL) return -1;

    GetImageEncoders(num, size, pImageCodecInfo);

    for (UINT j = 0; j < num; ++j) {
        if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
            *pClsid = pImageCodecInfo[j].Clsid;
            free(pImageCodecInfo);
            return j;
        }
    }

    free(pImageCodecInfo);
    return -1;
}

LRESULT CALLBACK ScreenCapture::OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    ScreenCapture* pThis = reinterpret_cast<ScreenCapture*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

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

            pThis->selectedRect.left = min(pThis->startPoint.x, pThis->endPoint.x);
            pThis->selectedRect.top = min(pThis->startPoint.y, pThis->endPoint.y);
            pThis->selectedRect.right = max(pThis->startPoint.x, pThis->endPoint.x);
            pThis->selectedRect.bottom = max(pThis->startPoint.y, pThis->endPoint.y);

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
                HPEN oldPen = (HPEN)SelectObject(hdc, pen);

                Rectangle(hdc,
                    min(pThis->startPoint.x, pThis->endPoint.x),
                    min(pThis->startPoint.y, pThis->endPoint.y),
                    max(pThis->startPoint.x, pThis->endPoint.x),
                    max(pThis->startPoint.y, pThis->endPoint.y));

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
    const wchar_t* className = L"ScreenCaptureOverlay";

    WNDCLASSW wc = {};
    wc.lpfnWndProc = OverlayWndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(NULL, IDC_CROSS);
    wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);

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

// 捕获屏幕区域
bool ScreenCapture::CaptureScreenRegion(const RECT& rect, const std::wstring& filename) {
    int width = rect.right - rect.left;
    int height = rect.bottom - rect.top;

    if (width <= 0 || height <= 0) {
        return false;
    }

    // 获取屏幕DC
    HDC screenDC = GetDC(NULL);
    HDC memDC = CreateCompatibleDC(screenDC);
    HBITMAP bitmap = CreateCompatibleBitmap(screenDC, width, height);
    HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, bitmap);

    // 复制屏幕内容到位图
    BitBlt(memDC, 0, 0, width, height, screenDC, rect.left, rect.top, SRCCOPY);

    // 使用GDI+保存为PNG
    Bitmap* gdipBitmap = new Bitmap(bitmap, NULL);

    CLSID pngClsid;
    if (GetEncoderClsid(L"image/png", &pngClsid) >= 0) {
        gdipBitmap->Save(filename.c_str(), &pngClsid, NULL);
    }

    delete gdipBitmap;
    SelectObject(memDC, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(memDC);
    ReleaseDC(NULL, screenDC);

    return true;
}

// 开始截图选择
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

// 获取选择的区域
RECT ScreenCapture::GetSelectedRect() const {
    return selectedRect;
}