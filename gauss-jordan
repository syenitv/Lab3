#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

static double now_seconds(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

static double** alloc_2d(int rows, int cols) {
    double** M = (double**)malloc((size_t)rows * sizeof(double*));
    M[0] = (double*)malloc((size_t)rows * cols * sizeof(double));
    for (int i = 1; i < rows; i++) M[i] = M[0] + (size_t)i * cols;
    return M;
}
static void free_2d(double** M) { free(M[0]); free(M); }

static void fill_random(double** A, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
}

static int invert_gauss_jordan(double** A, double** Inv, int n) {
    const int W = 2 * n;
    double** aug = alloc_2d(n, W);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
        for (int j = 0; j < n; j++) aug[i][n + j] = (i == j) ? 1.0 : 0.0;
    }

    for (int k = 0; k < n; k++) {
        int piv = k;
        double maxv = fabs(aug[k][k]);
        for (int i = k + 1; i < n; i++) {
            double v = fabs(aug[i][k]);
            if (v > maxv) { maxv = v; piv = i; }
        }
        if (maxv < 1e-12) { free_2d(aug); return 0; }

        if (piv != k)
            for (int j = 0; j < W; j++) {
                double t = aug[k][j]; aug[k][j] = aug[piv][j]; aug[piv][j] = t;
            }

        double d = aug[k][k];
        for (int j = 0; j < W; j++) aug[k][j] /= d;

        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            double f = aug[i][k];
            if (fabs(f) < 1e-15) continue;
            for (int j = 0; j < W; j++)
                aug[i][j] -= f * aug[k][j];
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) Inv[i][j] = aug[i][n + j];

    free_2d(aug);
    return 1;
}

static double frob_error(double** A, double** Inv, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double v = 0.0;
            for (int k = 0; k < n; k++) v += A[i][k] * Inv[k][j];
            if (i == j) v -= 1.0;
            s += v * v;
        }
    return sqrt(s) / n;
}

int main(void) {
    srand(12345);

    printf("Enter matrix sizes one by one (0 to stop):\n");
    printf("%6s | %14s | %12s\n",
           "n", "invert, s", "err_F / n");

    for (;;) {
        int n = 0;
        printf("n = ");
        if (scanf("%d", &n) != 1) {
            fprintf(stderr, "Invalid input, stopping.\n");
            break;
        }
        if (n == 0) break;
        if (n < 0) {
            fprintf(stderr, "Size must be positive, try again.\n");
            continue;
        }

        double** A = alloc_2d(n, n);
        double** Inv = alloc_2d(n, n);
        fill_random(A, n);

        double t0 = now_seconds();
        int ok = invert_gauss_jordan(A, Inv, n);
        double t1 = now_seconds();

        if (!ok) {
            printf("%6d | %14.6f | singular\n", n, t1 - t0);
        }
        else {
            double err = frob_error(A, Inv, n);
            printf("%6d | %14.6f | %12.3e\n", n, t1 - t0, err);
        }

        free_2d(A);
        free_2d(Inv);
    }

    return 0;
}
