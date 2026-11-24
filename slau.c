#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

static double now_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER now;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static void print_matrix(const char* title, const float* mat, int n) {
    printf("\n%s\n", title);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.4f ", mat[i * n + j]);
        }
        printf("\n");
    }
}

static void invert_gauss_jordan_cpu(float* A, float* Inv, int n) {
    int W = 2 * n;
    float* aug = (float*)malloc(n * W * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * W + j] = A[i * n + j];
        for (int j = 0; j < n; j++) aug[i * W + (n + j)] = (i == j) ? 1.0f : 0.0f;
    }
    for (int k = 0; k < n; k++) {
        float maxv = fabsf(aug[k * W + k]);
        int piv = k;
        for (int i = k + 1; i < n; i++) {
            if (fabsf(aug[i * W + k]) > maxv) {
                maxv = fabsf(aug[i * W + k]);
                piv = i;
            }
        }
        if (maxv < 1e-12f) continue;
        if (piv != k) {
            for (int j = 0; j < W; j++) {
                float t = aug[k * W + j];
                aug[k * W + j] = aug[piv * W + j];
                aug[piv * W + j] = t;
            }
        }
        float div = aug[k * W + k];
        for (int j = 0; j < W; j++) aug[k * W + j] /= div;
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            float f = aug[i * W + k];
            for (int j = 0; j < W; j++) aug[i * W + j] -= f * aug[k * W + j];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) Inv[i * n + j] = aug[i * W + (n + j)];
    }
    free(aug);
}

int main() {
    int n;
    printf("Enter matrix size N (e.g., 8, 128, 256, 512): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid size.\n");
        return 1;
    }
    float* A = (float*)malloc(n * n * sizeof(float));
    float* Inv = (float*)malloc(n * n * sizeof(float));
    srand(42);
    for (int i = 0; i < n * n; i++) A[i] = (float)rand() / RAND_MAX;

    double t0 = now_ms();
    invert_gauss_jordan_cpu(A, Inv, n);
    double t1 = now_ms();
    printf("time: %.2f ms\n", t1 - t0);

    print_matrix("Original Matrix A", A, n);
    print_matrix("Inverse Matrix A^-1 (CPU)", Inv, n);

    free(A); free(Inv);
    return 0;
}