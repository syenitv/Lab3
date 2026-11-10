#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <CL/cl.h>

const char* kernel_source =
"__kernel void jacobi(__global const float* A, \n"
"                     __global const float* b, \n"
"                     __global const float* x, \n"
"                     __global float* x_new, \n"
"                     const int n) \n"
"{ \n"
"    int i = get_global_id(0); \n"
"    if (i >= n) return; \n"
"    float s = b[i]; \n"
"    for (int j = 0; j < n; ++j) { \n"
"        if (j != i) s -= A[i * n + j] * x[j]; \n"
"    } \n"
"    x_new[i] = s / A[i * n + i]; \n"
"} \n";

static double now_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER now;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static unsigned long long lcg_state = 42;
static double lcg_double(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int u = (unsigned int)(lcg_state >> 33);
    return (double)u / (double)0x7FFFFFFF - 1.0;
}

static void generate_slau(int n, float* A, float* b, float* x_true) {
    for (int i = 0; i < n; ++i) x_true[i] = (float)lcg_double();
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i == j) { A[i * n + j] = 0.0f; continue; }
            float v = (float)lcg_double();
            A[i * n + j] = v;
            row_sum += fabsf(v);
        }
        A[i * n + i] = row_sum + 1.0f + fabsf((float)lcg_double());
    }
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += A[i * n + j] * x_true[j];
        b[i] = s;
    }
}

static void jacobi_cpu(const float* A, const float* b, float* x, int n, int max_iter) {
    float* x_new = (float*)malloc(n * sizeof(float));
    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < n; ++i) {
            float s = b[i];
            for (int j = 0; j < n; ++j) if (j != i) s -= A[i * n + j] * x[j];
            x_new[i] = s / A[i * n + i];
        }
        memcpy(x, x_new, n * sizeof(float));
    }
    free(x_new);
}

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
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Kernel compilation error:\n%s\n", log);
        return 1;
    }
    kernel = clCreateKernel(program, "jacobi", &err);
    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    int max_iter = 5000;
    printf("n,iters,GPU_time_ms,CPU_time_ms,max_err_gpu_vs_cpu\n");
    for (int k = 0; k < 5; ++k) {
        int n = sizes[k];
        size_t n_bytes = n * sizeof(float);
        size_t mat_bytes = n * n * sizeof(float);
        float* A = (float*)malloc(mat_bytes);
        float* b = (float*)malloc(n_bytes);
        float* x_true = (float*)malloc(n_bytes);
        float* x_gpu = (float*)calloc(n, sizeof(float));
        float* x_cpu = (float*)calloc(n, sizeof(float));
        generate_slau(n, A, b, x_true);
        cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_bytes, A, &err);
        cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bytes, b, &err);
        cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bytes, x_gpu, &err);
        cl_mem d_x_new = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_bytes, NULL, &err);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_x);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x_new);
        clSetKernelArg(kernel, 4, sizeof(int), &n);
        size_t global_work_size = n;
        double t0_gpu = now_ms();
        for (int it = 0; it < max_iter; ++it) {
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            cl_mem temp = d_x;
            d_x = d_x_new;
            d_x_new = temp;
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_x);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x_new);
        }
        clFinish(queue);
        double t1_gpu = now_ms();
        clEnqueueReadBuffer(queue, d_x, CL_TRUE, 0, n_bytes, x_gpu, 0, NULL, NULL);
        double t0_cpu = now_ms();
        jacobi_cpu(A, b, x_cpu, n, max_iter);
        double t1_cpu = now_ms();
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = fabsf(x_gpu[i] - x_cpu[i]);
            if (diff > max_err) max_err = diff;
        }
        printf("%d,%d,%.2f,%.2f,%.3e\n", n, max_iter, t1_gpu - t0_gpu, t1_cpu - t0_cpu, max_err);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_x);
        clReleaseMemObject(d_x_new);
        free(A); free(b); free(x_true); free(x_gpu); free(x_cpu);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}