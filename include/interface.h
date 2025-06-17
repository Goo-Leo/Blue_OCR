//
// Created by 10633 on 2025/6/12.
//

#ifndef INTERFACE_H
#define INTERFACE_H

#include <windows.h>
#include <string>
#include <opencv2/opencv.hpp>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")


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


// class Show {
//     private:
//     void show_result();
// };

#endif //INTERFACE_H
