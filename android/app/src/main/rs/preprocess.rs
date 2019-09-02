#pragma version(1)
#pragma rs java_package_name(com.sicariusnoctis.collaborativeintelligence)
#pragma rs_fp_relaxed

/*
uchar4 RS_KERNEL preprocess(uchar4 in, uint32_t x, uint32_t y) {
    uchar4 out = in;
    return out;
}
*/

/*
uint32_t RS_KERNEL preprocess(uchar4 in, uint32_t x, uint32_t y) {
    uint32_t out = in.a << 24 | in.r << 16 | in.g << 8 | in.b;
    return out;
}
*/

uchar4 RS_KERNEL preprocess(uchar4 in, uint32_t x, uint32_t y) {
    uchar4 out;
    out[0] = in.a;
    out[1] = in.r;
    out[2] = in.g;
    out[3] = in.b;
    return out;
}

/*
rs_allocation gInputFrame;
rs_allocation gOutputFrame;

uint32_t xStart, yStart;
uint32_t outputWidth, outputHeight;

uchar4 RS_KERNEL preprocess(uchar4 in, uint32_t x, uint32_t y) {
    uchar Y = rsGetElementAtYuv_uchar_Y(gInputFrame, x, y);
    uchar U = rsGetElementAtYuv_uchar_U(gInputFrame, x, y);
    uchar V = rsGetElementAtYuv_uchar_V(gInputFrame, x, y);

    uchar4 rgba = rsYuvToRGBA_uchar4(Y, U, V);
    rgba.a = 0xFF;

    uint32_t translated_x = x - xStart;
    uint32_t translated_y = y - yStart;

    rsSetElementAt_uchar4(gOutputFrame, rgba, translated_x, translated_y);
    return rgba;
}
*/