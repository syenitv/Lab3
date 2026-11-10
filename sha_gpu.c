#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <openssl/sha.h>
#include <CL/cl.h>

const char* kernel_source =
"inline ulong ROTR(ulong x, int n) { return (x >> n) | (x << (64 - n)); }\n"
"inline ulong CH(ulong x, ulong y, ulong z) { return (x & y) ^ (~x & z); }\n"
"inline ulong MAJ(ulong x, ulong y, ulong z) { return (x & y) ^ (x & z) ^ (y & z); }\n"
"inline ulong SIGMA0(ulong x) { return ROTR(x, 28) ^ ROTR(x, 34) ^ ROTR(x, 39); }\n"
"inline ulong SIGMA1(ulong x) { return ROTR(x, 14) ^ ROTR(x, 18) ^ ROTR(x, 41); }\n"
"inline ulong sigma0(ulong x) { return ROTR(x, 1) ^ ROTR(x, 8) ^ (x >> 7); }\n"
"inline ulong sigma1(ulong x) { return ROTR(x, 19) ^ ROTR(x, 61) ^ (x >> 6); }\n"
"\n"
"__kernel void brute_sha384(__global const uchar* target, \n"
"                           __global int* found_idx, \n"
"                           const ulong total_combinations, \n"
"                           __global const char* alphabet) \n"
"{ \n"
"    ulong id = get_global_id(0); \n"
"    if (id >= total_combinations || *found_idx != -1) return; \n"
"\n"
"    uchar msg[128] = {0}; \n"
"    ulong temp = id; \n"
"    for (int i = 4; i >= 0; --i) { msg[i] = alphabet[temp % 36]; temp /= 36; } \n"
"    msg[5] = 0x80; \n"
"    msg[126] = 0x00; msg[127] = 40; \n"
"\n"
"    ulong K[80] = { \n"
"        0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL, 0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL, \n"
"        0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL, 0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL, \n"
"        0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL, 0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL, \n"
"        0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL, 0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL, \n"
"        0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL, 0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL, \n"
"        0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL, 0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL, \n"
"        0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL, 0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL, \n"
"        0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL, 0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL, \n"
"        0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL, 0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL, \n"
"        0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL, 0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL \n"
"    }; \n"
"\n"
"    ulong H[8] = { \n"
"        0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL, 0x9159015a3070dd17UL, 0x152fecd8f70e5939UL, \n"
"        0x67332667ffc00b31UL, 0x8eb44a8768581511UL, 0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL \n"
"    }; \n"
"\n"
"    ulong W[80]; \n"
"    for (int i = 0; i < 16; ++i) { \n"
"        W[i] = ((ulong)msg[i*8] << 56) | ((ulong)msg[i*8+1] << 48) | ((ulong)msg[i*8+2] << 40) | ((ulong)msg[i*8+3] << 32) | \n"
"               ((ulong)msg[i*8+4] << 24) | ((ulong)msg[i*8+5] << 16) | ((ulong)msg[i*8+6] << 8) | (ulong)msg[i*8+7]; \n"
"    } \n"
"    for (int i = 16; i < 80; ++i) { W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]; } \n"
"\n"
"    ulong a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7]; \n"
"    for (int i = 0; i < 80; ++i) { \n"
"        ulong t1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + W[i]; \n"
"        ulong t2 = SIGMA0(a) + MAJ(a, b, c); \n"
"        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2; \n"
"    } \n"
"\n"
"    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h; \n"
"\n"
"    int match = 1; \n"
"    for (int i = 0; i < 6; ++i) { \n"
"        uchar h_byte = (uchar)(H[i] >> 56); if (h_byte != target[i*8]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 48); if (h_byte != target[i*8+1]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 40); if (h_byte != target[i*8+2]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 32); if (h_byte != target[i*8+3]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 24); if (h_byte != target[i*8+4]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 16); if (h_byte != target[i*8+5]) match = 0; \n"
"        h_byte = (uchar)(H[i] >> 8);  if (h_byte != target[i*8+6]) match = 0; \n"
"        h_byte = (uchar)(H[i]);       if (h_byte != target[i*8+7]) match = 0; \n"
"    } \n"
"    if (match) { *found_idx = id; } \n"
"} \n";

int main() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU not found, using CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Kernel compilation error:\n%s\n", log);
        return 1;
    }
    kernel = clCreateKernel(program, "brute_sha384", &err);
    const char* alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
    uint64_t N = 60466176;
    int found_idx = -1;
    unsigned char target[48];
    const char* real = "abc12";
    SHA384((const unsigned char*)real, strlen(real), target);
    cl_mem d_alphabet = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 37, (void*)alphabet, &err);
    cl_mem d_target = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 48, target, &err);
    cl_mem d_found = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &found_idx, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_target);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_found);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &N);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_alphabet);
    size_t global_work_size = N;
    printf("GPU brute-forcing %llu combinations...\n", (unsigned long long)global_work_size);
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    QueryPerformanceCounter(&end);
    double time_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    clEnqueueReadBuffer(queue, d_found, CL_TRUE, 0, sizeof(int), &found_idx, 0, NULL, NULL);
    printf("Finished in %.2f ms!\n", time_ms);
    printf("Speed: %.0f hashes per second\n", N / (time_ms / 1000.0));
    if (found_idx != -1) {
        char found_str[6];
        uint64_t temp = found_idx;
        for (int i = 4; i >= 0; --i) {
            found_str[i] = alphabet[temp % 36];
            temp /= 36;
        }
        found_str[5] = '\0';
        printf("FOUND PASSWORD: %s (at index %d)\n", found_str, found_idx);
    }
    else {
        printf("Password not found.\n");
    }
    clReleaseMemObject(d_alphabet);
    clReleaseMemObject(d_target);
    clReleaseMemObject(d_found);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}