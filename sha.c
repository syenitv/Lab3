#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <openssl/sha.h>

static double now_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER now;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static const char* ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";
#define ALPHA_LEN 36

static void make_candidate(uint64_t idx, char* buf, int len) {
    for (int i = len - 1; i >= 0; --i) {
        buf[i] = ALPHABET[idx % ALPHA_LEN];
        idx /= ALPHA_LEN;
    }
    buf[len] = '\0';
}

static void to_hex(const unsigned char* d, int n, char* out) {
    for (int i = 0; i < n; ++i) sprintf(out + i * 2, "%02x", d[i]);
    out[n * 2] = '\0';
}

int main(void) {
    unsigned char t[SHA384_DIGEST_LENGTH];
    char hex[2 * SHA384_DIGEST_LENGTH + 1];
    SHA384((const unsigned char*)"abc", 3, t);
    to_hex(t, SHA384_DIGEST_LENGTH, hex);
    printf("self-test SHA384(\"abc\") = %s\n", hex);
    printf("expected  = ""cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed""8086072ba1e7cc2358baeca134c825a7\n\n");
    const char* real = "abc12";
    unsigned char target[SHA384_DIGEST_LENGTH];
    SHA384((const unsigned char*)real, strlen(real), target);
    printf("len,N,checked,found,time_ms,hashes_per_sec\n");
    for (int len = 3; len <= 5; ++len) {
        uint64_t N = 1;
        for (int i = 0; i < len; ++i) N *= ALPHA_LEN;
        char cand[16], found[16] = { 0 };
        uint64_t checked = 0;
        double t0 = now_ms();
        for (uint64_t i = 0; i < N; ++i) {
            make_candidate(i, cand, len);
            unsigned char h[SHA384_DIGEST_LENGTH];
            SHA384((const unsigned char*)cand, strlen(cand), h);
            ++checked;
            if (memcmp(h, target, SHA384_DIGEST_LENGTH) == 0) {
                strcpy(found, cand);
                break;
            }
        }
        double t1 = now_ms();
        double ms = t1 - t0;
        printf("%d,%llu,%llu,%s,%.1f,%.0f\n",
            len, (unsigned long long)N, (unsigned long long)checked,
            found[0] ? found : "(not found)", ms,
            checked / (ms / 1000.0));
    }
    return 0;
}