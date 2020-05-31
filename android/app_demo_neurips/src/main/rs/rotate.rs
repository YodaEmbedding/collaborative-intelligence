#pragma version(1)
#pragma rs java_package_name(io.github.yodaembedding.collaborativeintelligence)
#pragma rs_fp_relaxed

rs_allocation input;
uint32_t width;
uint32_t height;

uchar4 RS_KERNEL rotate90(uint32_t x, uint32_t y) {
    const uchar4 *out = rsGetElementAt(input, y, width - 1 - x);
    return *out;
}

uchar4 RS_KERNEL rotate270(uint32_t x, uint32_t y) {
    const uchar4 *out = rsGetElementAt(input, height - 1 - y, x);
    return *out;
}

uchar4 RS_KERNEL rotate180(uint32_t x, uint32_t y) {
    const uchar4 *out = rsGetElementAt(input, width - x - 1, height - y - 1);
    return *out;
}