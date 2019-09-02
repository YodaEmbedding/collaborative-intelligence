#pragma version(1)
#pragma rs java_package_name(com.sicariusnoctis.collaborativeintelligence)
#pragma rs_fp_relaxed

rs_allocation input;
uint32_t xStart, yStart;

uchar4 RS_KERNEL crop(uint32_t x, uint32_t y) {
    return rsGetElementAt_uchar4(input, x + xStart, y + yStart);
}