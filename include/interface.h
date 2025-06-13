//
// Created by 10633 on 2025/6/12.
//

#ifndef INTERFACE_H
#define INTERFACE_H

#include <windows.h>
#include <gdiplus.h>
#include <string>

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

using namespace Gdiplus;

class ScreenCapture {
private:
    HWND overlayWnd;
    bool isSelecting;
    POINT startPoint;
    POINT endPoint;
    RECT selectedRect;

    // 私有辅助函数
    int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
    bool CreateOverlayWindow();

public:

    ScreenCapture();

    ~ScreenCapture();

    static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    bool StartCapture();
    bool CaptureScreenRegion(const RECT& rect, const std::wstring& filename);
    RECT GetSelectedRect() const;
};

#endif //INTERFACE_H
