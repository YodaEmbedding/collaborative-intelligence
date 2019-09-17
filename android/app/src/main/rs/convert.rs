#pragma version(1)
#pragma rs java_package_name(com.sicariusnoctis.collaborativeintelligence)
#pragma rs_fp_relaxed

rs_allocation output;
uint32_t width;

void RS_KERNEL rgba2rgb(uchar4 in, uint32_t x, uint32_t y) {
    uint32_t pos = 3 * (x + width * y);
    rsSetElementAt_uchar(output, in.r, pos);
    rsSetElementAt_uchar(output, in.g, pos + 1);
    rsSetElementAt_uchar(output, in.b, pos + 2);
}

void RS_KERNEL rgba2rgbFloat(uchar4 in, uint32_t x, uint32_t y) {
    uint32_t pos = 3 * (x + width * y);
    rsSetElementAt_float(output, in.r, pos);
    rsSetElementAt_float(output, in.g, pos + 1);
    rsSetElementAt_float(output, in.b, pos + 2);
}