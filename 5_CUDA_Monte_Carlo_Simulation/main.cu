// Monte Carlo simulation of castle-bombardment

// system includes
#define _USE_MATH_DEFINES
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

// setting the number of trials in the monte carlo simulation
#ifndef NUMTRIALS
#define NUMTRIALS (8 * 1024 * 1024)
#endif

// number of threads per block
#ifndef BLOCKSIZE
#define BLOCKSIZE 64
#endif

// number of blocks
#define NUMBLOCKS (NUMTRIALS / BLOCKSIZE)

// better to define these here so that the rand() calls don't get into the thread timing
float hvs[NUMTRIALS];
float hths[NUMTRIALS];
float hgs[NUMTRIALS];
float hhs[NUMTRIALS];
float hds[NUMTRIALS];
int hhits[NUMTRIALS];

// ranges for the random numbers
const float GMIN = 20.0; // ground distance in meters
const float GMAX = 30.0; // ground distance in meters
const float HMIN = 10.0; // cliff height in meters
const float HMAX = 20.0; // cliff height in meters
const float DMIN = 10.0; // distance to castle in meters
const float DMAX = 20.0; // distance to castle in meters
const float VMIN = 10.0; // intial cnnonball velocity in meters / sec
const float VMAX = 30.0; // intial cnnonball velocity in meters / sec
const float THMIN = 70.0; // cannonball launch angle in degrees
const float THMAX = 80.0; // cannonball launch angle in degrees

// constants
#define GRAVITY -9.8f // acceleraion due to gravity in meters / sec^2
#define TOL 5.0f // tolerance in cannonball hitting the castle in meters

// function prototypes
void CudaCheckError();
float Ranf(float, float);
void TimeOfDaySeed();

// degrees-to-radians -- callable from the device
__device__ float
Radians(float d)
{
    return (M_PI / 180.f) * d;
}

// the kernel
__global__ void
MonteCarlo(float* dvs, float* dths, float* dgs, float* dhs, float* dds, int* dhits)
{
    // unsigned int numItems = blockDim.x;
    // unsigned int tnum = threadIdx.x;
    // unsigned int wgNum = blockIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // randomize everything
    float v = dvs[gid];
    float thr = Radians(dths[gid]);
    float vx = v * cos(thr);
    float vy = v * sin(thr);
    float g = dgs[gid];
    float h = dhs[gid];
    float d = dds[gid];

    dhits[gid] = 0;

    // see if the ball doesn't even reach the cliff
    float t = -vy / (0.5 * GRAVITY);
    float x = vx * t;
    if (x > g) {
        // see if the ball hits the vertical cliff face
        t = g / vx;
        float y = vy * t + 0.5 * GRAVITY * t * t;

        if (y > h) {
            // the ball hits the upper deck
            float a = 0.5 * GRAVITY;
            float b = vy;
            float c = -h;
            float disc = b * b - 4.f * a * c; // quadratic formula discriminant

            // successfully hits the ground above the cliff
            // get the intersection
            disc = sqrtf(disc);
            float t1 = (-b + disc) / (2.f * a); // time to intersect high ground
            float t2 = (-b - disc) / (2.f * a); // time to intersect high ground
            float tmax = t1;
            if (t2 > tmax)
                tmax = t2; // only care about the second intersection

            // how far the ball lands horizontally from the edge of the cliff
            float upperDist = vx * tmax - g;

            // see if the ball hits the castle
            if (fabsf(upperDist - d) <= TOL) {
                dhits[gid] = 1;
            }
        } // if ball clears the cliff face
    } // if ball gets as far as the cliff face
}

int main(int argc, char* argv[])
{
    TimeOfDaySeed();

    // int dev = findCudaDevice(argc, (const char**)argv);

    // fill the random-value arrays
    for (int n = 0; n < NUMTRIALS; n++) {
        hvs[n] = Ranf(VMIN, VMAX);
        hths[n] = Ranf(THMIN, THMAX);
        hgs[n] = Ranf(GMIN, GMAX);
        hhs[n] = Ranf(HMIN, HMAX);
        hds[n] = Ranf(DMIN, DMAX);
    }

    // allocate device memory
    float *dvs, *dths, *dgs, *dhs, *dds;
    int* dhits;

    cudaMalloc(&dvs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dths, NUMTRIALS * sizeof(float));
    cudaMalloc(&dgs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dds, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhits, NUMTRIALS * sizeof(int));
    CudaCheckError();

    // copy host memory to the device
    cudaMemcpy(dvs, hvs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dths, hths, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgs, hgs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dhs, hhs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dds, hds, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    CudaCheckError();

    // setup the execution parameters
    dim3 grid(NUMBLOCKS, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // allocate cuda events that we'll use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CudaCheckError();

    // let the gpu go quiet
    cudaDeviceSynchronize();

    // record the start event
    cudaEventRecord(start, NULL);
    CudaCheckError();

    // execute the kernel
    MonteCarlo<<<grid, threads>>>(dvs, dths, dgs, dhs, dds, dhits);

    // record the stop event
    cudaEventRecord(stop, NULL);
    CudaCheckError();

    // wait for the stop event to complete
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    CudaCheckError();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    CudaCheckError();

    // compute and print the performance
    double secondsTotal = 0.001 * (double)msecTotal;
    double trialsPerSecond = (float)NUMTRIALS / secondsTotal;
    double megaTrialsPerSecond = trialsPerSecond / 1000000.;

    // copy result from the device to the host
    cudaMemcpy(hhits, dhits, NUMTRIALS * sizeof(int), cudaMemcpyDeviceToHost);
    CudaCheckError();

    // compute the sum
    int numHits = 0;
    for (int i = 0; i < NUMTRIALS; i++) {
        numHits += hhits[i];
    }

    // compute the probability
    float probability = 100.f * (float)numHits / (float)NUMTRIALS;

#define CSV
#ifdef CSV
    fprintf(stderr, "%10d , %5d , %8.2lf , %6.3f%%\n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, probability);
#else
    fprintf(stderr, "Trials = %10d, BlockSize = %5d, MegaTrials/Second = %8.2lf, Probability=%6.3f%%\n",
        NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, probability);
#endif

    // clean up device memory
    cudaFree(dvs);
    cudaFree(dths);
    cudaFree(dgs);
    cudaFree(dhs);
    cudaFree(dds);
    cudaFree(dhits);
    CudaCheckError();

    return 0;
}

void CudaCheckError()
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    }
}

float Ranf(float low, float high)
{
    float r = (float)rand(); // 0 - RAND_MAX
    float t = r / (float)RAND_MAX; // 0. - 1.
    return low + t * (high - low);
}

void TimeOfDaySeed()
{
    time_t now;
    time(&now);
    struct tm n = *localtime(&now);

    struct tm jan01 = *localtime(&now);
    jan01.tm_mon = 0;
    jan01.tm_mday = 1;
    jan01.tm_hour = 0;
    jan01.tm_min = 0;
    jan01.tm_sec = 0;

    double seconds = difftime(now, mktime(&jan01));
    unsigned int seed = (unsigned int)(1000. * seconds); // milliseconds
    srand(seed);
}
