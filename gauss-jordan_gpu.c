#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <CL/cl.h>

static const char* KERNEL_SRC =
"__kernel void k_pivot_swap(__global float* aug, int n, int k, int W, float eps, __global float* maxv_out) {\n"
"    int tid = get_local_id(0); int lsz = get_local_size(0);\n"
"    __local float sval[256]; __local int sidx[256];\n"
"    float lmax = 0.0f; int lidx = k;\n"
"    for (int i = k + tid; i < n; i += lsz) {\n"
"        float v = fabs(aug[(long)i * W + k]);\n"
"        if (v > lmax) { lmax = v; lidx = i; }\n"
"    }\n"
"    sval[tid] = lmax; sidx[tid] = lidx;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = lsz / 2; s > 0; s >>= 1) {\n"
"        if (tid < s && sval[tid + s] > sval[tid]) {\n"
"            sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    float maxv = sval[0]; int piv = sidx[0];\n"
"    if (tid == 0) *maxv_out = maxv;\n"
"    if (maxv < eps || piv == k) return;\n"
"    for (int j = tid; j < W; j += lsz) {\n"
"        float t = aug[(long)k * W + j];\n"
"        aug[(long)k * W + j] = aug[(long)piv * W + j];\n"
"        aug[(long)piv * W + j] = t;\n"
"    }\n"
"}\n"
"__kernel void k_normalize(__global float* aug, int k, int W) {\n"
"    int j = get_global_id(0); if (j >= W) return;\n"
"    aug[(long)k * W + j] /= aug[(long)k * W + k];\n"
"}\n"
"__kernel void k_eliminate(__global float* aug, int n, int k, int W, __local float* srow) {\n"
"    int i = get_group_id(0); int tid = get_local_id(0); int lsz = get_local_size(0);\n"
"    if (i == k || i >= n) return;\n"
"    for (int j = tid; j < W; j += lsz) srow[j] = aug[(long)k * W + j];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    float f = aug[(long)i * W + k];\n"
"    if (fabs(f) < 1e-15f) return;\n"
"    for (int j = tid; j < W; j += lsz) aug[(long)i * W + j] -= f * srow[j];\n"
"}\n"
"__kernel void k_extract(__global const float* aug, __global float* inv, int n, int W) {\n"
"    int i = get_global_id(0); int j = get_global_id(1);\n"
"    if (i >= n || j >= n) return;\n"
"    inv[(long)i * n + j] = aug[(long)i * W + (n + j)];\n"
"}\n";

static double now_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER now;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main() {
    int n;
    printf("Enter matrix size N for GPU (e.g., 128, 256, 512, 1024): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid size.\n");
        return 1;
    }
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_piv, k_norm, k_elim, k_ext;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU not found.\n");
        return 1;
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &KERNEL_SRC, NULL, &err);
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Kernel compilation error:\n%s\n", log);
        return 1;
    }
    k_piv = clCreateKernel(program, "k_pivot_swap", &err);
    k_norm = clCreateKernel(program, "k_normalize", &err);
    k_elim = clCreateKernel(program, "k_eliminate", &err);
    k_ext = clCreateKernel(program, "k_extract", &err);
    int W = 2 * n;
    float* h_aug = (float*)malloc(n * W * sizeof(float));
    srand(42);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) h_aug[i * W + j] = (float)rand() / RAND_MAX;
        for (int j = 0; j < n; j++) h_aug[i * W + (n + j)] = (i == j) ? 1.0f : 0.0f;
    }
    cl_mem d_aug = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * W * sizeof(float), h_aug, &err);
    cl_mem d_inv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * n * sizeof(float), NULL, &err);
    cl_mem d_maxv = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    size_t T = 256;
    size_t gP = T, lP = T;
    size_t gN = ((W + T - 1) / T) * T, lN = T;
    size_t gE = (size_t)n * T, lE = T;
    size_t gsize[2] = { (size_t)n, (size_t)n };
    printf("Calculating inverse matrix for N=%d on GPU...\n", n);
    double t0 = now_ms();
    for (int k = 0; k < n; k++) {
        clSetKernelArg(k_piv, 0, sizeof(cl_mem), &d_aug);
        clSetKernelArg(k_piv, 1, sizeof(int), &n);
        clSetKernelArg(k_piv, 2, sizeof(int), &k);
        clSetKernelArg(k_piv, 3, sizeof(int), &W);
        float eps = 1e-12f;
        clSetKernelArg(k_piv, 4, sizeof(float), &eps);
        clSetKernelArg(k_piv, 5, sizeof(cl_mem), &d_maxv);
        clEnqueueNDRangeKernel(queue, k_piv, 1, NULL, &gP, &lP, 0, NULL, NULL);
        clSetKernelArg(k_norm, 0, sizeof(cl_mem), &d_aug);
        clSetKernelArg(k_norm, 1, sizeof(int), &k);
        clSetKernelArg(k_norm, 2, sizeof(int), &W);
        clEnqueueNDRangeKernel(queue, k_norm, 1, NULL, &gN, &lN, 0, NULL, NULL);
        clSetKernelArg(k_elim, 0, sizeof(cl_mem), &d_aug);
        clSetKernelArg(k_elim, 1, sizeof(int), &n);
        clSetKernelArg(k_elim, 2, sizeof(int), &k);
        clSetKernelArg(k_elim, 3, sizeof(int), &W);
        clSetKernelArg(k_elim, 4, W * sizeof(float), NULL);
        clEnqueueNDRangeKernel(queue, k_elim, 1, NULL, &gE, &lE, 0, NULL, NULL);
    }
    clSetKernelArg(k_ext, 0, sizeof(cl_mem), &d_aug);
    clSetKernelArg(k_ext, 1, sizeof(cl_mem), &d_inv);
    clSetKernelArg(k_ext, 2, sizeof(int), &n);
    clSetKernelArg(k_ext, 3, sizeof(int), &W);
    clEnqueueNDRangeKernel(queue, k_ext, 2, NULL, gsize, NULL, 0, NULL, NULL);
    clFinish(queue);
    double t1 = now_ms();
    printf("GPU time: %.2f ms\n", t1 - t0);
    clReleaseMemObject(d_aug);
    clReleaseMemObject(d_inv);
    clReleaseMemObject(d_maxv);
    clReleaseKernel(k_piv); clReleaseKernel(k_norm); clReleaseKernel(k_elim); clReleaseKernel(k_ext);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_aug);
    return 0;
}
