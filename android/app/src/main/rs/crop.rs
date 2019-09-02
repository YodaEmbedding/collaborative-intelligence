#pragma version(1)
#pragma rs java_package_name(com.sicariusnoctis.collaborativeintelligence)
#pragma rs_fp_relaxed

rs_allocation output;
uint32_t xStart, yStart;

void RS_KERNEL crop(uchar4 in, uint32_t x, uint32_t y) {
    rsSetElementAt_uchar4(output, in, x - xStart, y - yStart);
}

/*
rs_allocation inputFrame;
rs_allocation outputFrame;

uint32_t xStart, yStart;
uint32_t outputWidth, outputHeight;

uchar4 RS_KERNEL crop(uchar4 in, uint32_t x, uint32_t y) {
    uchar Y = rsGetElementAtYuv_uchar_Y(inputFrame, x, y);
    uchar U = rsGetElementAtYuv_uchar_U(inputFrame, x, y);
    uchar V = rsGetElementAtYuv_uchar_V(inputFrame, x, y);

    uchar4 rgba = rsYuvToRGBA_uchar4(Y, U, V);
    rgba.a = 0xFF;

    rsSetElementAt_uchar4(outputFrame, rgba, x - xStart, y - yStart);
    return rgba;
}
*/