#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>

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

static void generate_slau(int n, double* A, double* b, double* x_true) {
    int i, j;
    for (i = 0; i < n; ++i) x_true[i] = lcg_double();
    for (i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (j = 0; j < n; ++j) {
            if (i == j) { A[(size_t)i * n + j] = 0.0; continue; }
            double v = lcg_double();
            A[(size_t)i * n + j] = v;
            row_sum += fabs(v);
        }
        A[(size_t)i * n + i] = row_sum + 1.0 + fabs(lcg_double());
    }
    for (i = 0; i < n; ++i) {
        double s = 0.0;
        for (j = 0; j < n; ++j) s += A[(size_t)i * n + j] * x_true[j];
        b[i] = s;
    }
}

static int jacobi_cpu(const double* A, const double* b, double* x,
    int n, int max_iter, double eps) {
    double* x_new = (double*)malloc((size_t)n * sizeof(double));
    int it_done = 0, i, j;
    for (int it = 0; it < max_iter; ++it) {
        for (i = 0; i < n; ++i) {
            const double* row = A + (size_t)i * n;
            double s = b[i];
            for (j = 0; j < n; ++j) if (j != i) s -= row[j] * x[j];
            x_new[i] = s / row[i];
        }
        double diff = 0.0;
        for (i = 0; i < n; ++i) {
            double d = fabs(x_new[i] - x[i]);
            if (d > diff) diff = d;
        }
        memcpy(x, x_new, (size_t)n * sizeof(double));
        it_done = it + 1;
        if (diff < eps) break;
    }
    free(x_new);
    return it_done;
}

int main(void) {
    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    int ns = (int)(sizeof(sizes) / sizeof(sizes[0]));
    printf("n,iters,time_ms,max_err_vs_true\n");
    for (int k = 0; k < ns; ++k) {
        int n = sizes[k];
        double* A = (double*)malloc((size_t)n * n * sizeof(double));
        double* b = (double*)malloc((size_t)n * sizeof(double));
        double* x_true = (double*)malloc((size_t)n * sizeof(double));
        double* x = (double*)calloc((size_t)n, sizeof(double));
        generate_slau(n, A, b, x_true);
        double t0 = now_ms();
        int iters = jacobi_cpu(A, b, x, n, 5000, 1e-10);
        double t1 = now_ms();
        double err = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = fabs(x[i] - x_true[i]);
            if (d > err) err = d;
        }
        printf("%d,%d,%.3f,%.3e\n", n, iters, t1 - t0, err);
        free(A); free(b); free(x_true); free(x);
    }
    return 0;
}