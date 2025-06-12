//
// Created by 10633 on 2025/6/12.
//

#ifndef INTERFACE_H
#define INTERFACE_H

#include <Windows.h>
#include <iostream>

using namespace System;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace System::Windows::Forms;

class ScreenCapture {
private:
    static HWND selectionWindow;
    static RECT selectionRect;
    static bool isSelecting;
    static POINT startPoint;
    static POINT currentPoint;

    static bool CaptureRegion(int x, int y, int width, int height, const std::string& filename)
    {
        HDC hScreenDC = GetDC(NULL);
        HDC hMemDC = CreateCompatibleDC(hScreenDC);

        HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemDC, hBitmap);

        BitBlt(hMemDC, 0, 0, width, height, hScreenDC, x, y, SRCCOPY);

        bool result = SaveBitmapToFile(hBitmap, filename);

        SelectObject(hMemDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);

        return result;
    }
}

#endif //INTERFACE_H
