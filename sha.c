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

int main(void) {
    char word_input[128];
    unsigned char target[SHA384_DIGEST_LENGTH];
    printf("Enter a word to brute-force (3 to 5 characters):\n");
    scanf("%s", word_input);
    int input_len = strlen(word_input);
    if (input_len < 3 || input_len > 5) {
        printf("Error: Word length must be between 3 and 5 characters!\n");
        return 1;
    }
    SHA384((const unsigned char*)word_input, strlen(word_input), target);
    printf("\nSearching for word: %s\n", word_input);
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
                strcpy(found, cand); break;
            }
        }
        double ms = now_ms() - t0;
        printf("%d,%llu,%llu,%s,%.1f,%.0f\n", len, (unsigned long long)N, (unsigned long long)checked, found[0] ? found : "(not found)", ms, checked / (ms / 1000.0));
        for (int i = 0; i < SHA384_DIGEST_LENGTH; i++) printf("%llu", (unsigned long long)target[i]);
        if (found[0]) break;
    }
    return 0;
}