//
// Created by 10633 on 2025/6/12.
//

#ifndef INTERFACE_H
#define INTERFACE_H

#include <windows.h>
#include <shlwapi.h>
#include <opencv2/opencv.hpp>
#include "infer.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "Shlwapi.lib")

class ScreenCapture {
private:
    HWND overlayWnd;
    bool isSelecting;
    POINT startPoint;
    POINT endPoint;
    RECT selectedRect;

    int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
    bool CreateOverlayWindow();

public:
    ScreenCapture();

    ~ScreenCapture();

    static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    bool StartCapture();

    RECT GetSelectedRect() const;

    cv::Mat CaptureScreenRegion(const RECT& rect);
};


int show_result(const std::vector<rec_result> &texts);

#endif //INTERFACE_H
