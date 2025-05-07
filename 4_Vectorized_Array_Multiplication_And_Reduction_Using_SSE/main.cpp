#include <ctime>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

// Conditional include for Intel intrinsics
#ifdef USE_INTRINSICS
#include <immintrin.h>
#endif

// SSE stands for Streaming SIMD Extensions
#define SSE_WIDTH 4
#define ALIGNED __attribute__((aligned(16)))

#define NUMTRIES 100

#ifndef ARRAYSIZE
#define ARRAYSIZE 1024 * 1024
#endif

ALIGNED float A[ARRAYSIZE];
ALIGNED float B[ARRAYSIZE];
ALIGNED float C[ARRAYSIZE];

void SimdMul(float*, float*, float*, int);
void NonSimdMul(float*, float*, float*, int);
float SimdMulSum(float*, float*, int);
float NonSimdMulSum(float*, float*, int);

int main(int argc, char* argv[])
{
    // Output in CSV format instead of tab-separated format
    // Format: ArraySize,NonSimdMul,SimdMul,SpeedupMul,NonSimdMulSum,SimdMulSum,SpeedupMulSum

    // Variables for mul performance
    double nonSimdMulPerf = 0.0; // Non-SIMD multiplication performance
    double simdMulPerf = 0.0; // SIMD multiplication performance
    double mulSpeedup = 0.0; // Speedup ratio for multiplication

    // Variables for mulsum performance
    double nonSimdMulSumPerf = 0.0; // Non-SIMD multiplication+sum performance
    double simdMulSumPerf = 0.0; // SIMD multiplication+sum performance
    double mulSumSpeedup = 0.0; // Speedup ratio for multiplication+sum

    for (int i = 0; i < ARRAYSIZE; i++) {
        A[i] = sqrtf((float)(i + 1));
        B[i] = sqrtf((float)(i + 1));
    }

    // Test 1: Non-SIMD multiplication
    double maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        NonSimdMul(A, B, C, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance)
            maxPerformance = perf;
    }
    nonSimdMulPerf = maxPerformance / 1000000.;

    // Test 2: SIMD multiplication
    maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        SimdMul(A, B, C, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance)
            maxPerformance = perf;
    }
    simdMulPerf = maxPerformance / 1000000.;
    mulSpeedup = simdMulPerf / nonSimdMulPerf;

    // Test 3: Non-SIMD multiplication with sum
    maxPerformance = 0.;
    float sumn, sums;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        sumn = NonSimdMulSum(A, B, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance)
            maxPerformance = perf;
    }
    nonSimdMulSumPerf = maxPerformance / 1000000.;

    // Test 4: SIMD multiplication with sum
    maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        sums = SimdMulSum(A, B, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance)
            maxPerformance = perf;
    }
    simdMulSumPerf = maxPerformance / 1000000.;
    mulSumSpeedup = simdMulSumPerf / nonSimdMulSumPerf;

    // Output the CSV line: ArraySize,NonSimdMul,SimdMul,SpeedupMul,NonSimdMulSum,SimdMulSum,SpeedupMulSum
    fprintf(stderr, "%d,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf\n",
        ARRAYSIZE,
        nonSimdMulPerf,
        simdMulPerf,
        mulSpeedup,
        nonSimdMulSumPerf,
        simdMulSumPerf,
        mulSumSpeedup);

    return 0;
}

void NonSimdMul(float* A, float* B, float* C, int n)
{
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

float NonSimdMulSum(float* A, float* B, int n)
{
    float sum = 0.;
    for (int i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

void SimdMul(float* a, float* b, float* c, int len)
{
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;

#ifdef USE_INTRINSICS
    // Process 4 floats at a time using SSE
    for (int i = 0; i < limit; i += SSE_WIDTH) {
        // Load 4 floats from arrays a and b
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);

        // Multiply them together
        __m128 vc = _mm_mul_ps(va, vb);

        // Store the result in array c
        _mm_storeu_ps(&c[i], vc);
    }
#else
    __asm(
        ".att_syntax\n\t"
        "movq    -24(%rbp), %r8\n\t" // a
        "movq    -32(%rbp), %rcx\n\t" // b
        "movq    -40(%rbp), %rdx\n\t" // c
    );

    for (int i = 0; i < limit; i += SSE_WIDTH) {
        __asm(
            ".att_syntax\n\t"
            "movups	(%r8), %xmm0\n\t" // load the first sse register
            "movups	(%rcx), %xmm1\n\t" // load the second sse register
            "mulps	%xmm1, %xmm0\n\t" // do the multiply
            "movups	%xmm0, (%rdx)\n\t" // store the result
            "addq $16, %r8\n\t"
            "addq $16, %rcx\n\t"
            "addq $16, %rdx\n\t");
    }
#endif

    // Handle any remaining elements
    for (int i = limit; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}

float SimdMulSum(float* a, float* b, int len)
{
    ALIGNED float sum[4] = { 0., 0., 0., 0. };
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;

#ifdef USE_INTRINSICS
    // Initialize sum vector to zeros
    __m128 vsum = _mm_setzero_ps();

    // Process 4 floats at a time using SSE
    for (int i = 0; i < limit; i += SSE_WIDTH) {
        // Load 4 floats from arrays a and b
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);

        // Multiply them together
        __m128 vmul = _mm_mul_ps(va, vb);

        // Add to the running sum
        vsum = _mm_add_ps(vsum, vmul);
    }

    // Store the result in our sum array
    _mm_store_ps(sum, vsum);
#else
    __asm(
        ".att_syntax\n\t"
        "movq    -40(%rbp), %r8\n\t" // a
        "movq    -48(%rbp), %rcx\n\t" // b
        "leaq    -32(%rbp), %rdx\n\t" // &sum[0]
        "movups	 (%rdx), %xmm2\n\t" // 4 copies of 0. in xmm2
    );

    for (int i = 0; i < limit; i += SSE_WIDTH) {
        __asm(
            ".att_syntax\n\t"
            "movups	(%r8), %xmm0\n\t" // load the first sse register
            "movups	(%rcx), %xmm1\n\t" // load the second sse register
            "mulps	%xmm1, %xmm0\n\t" // do the multiply
            "addps	%xmm0, %xmm2\n\t" // do the add
            "addq $16, %r8\n\t"
            "addq $16, %rcx\n\t");
    }

    __asm(
        ".att_syntax\n\t"
        "movups	 %xmm2, (%rdx)\n\t" // copy the sums back to sum[ ]
    );
#endif

    // Handle any remaining elements
    for (int i = limit; i < len; i++) {
        sum[0] += a[i] * b[i];
    }

    return sum[0] + sum[1] + sum[2] + sum[3];
}
