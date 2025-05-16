#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cl.h"
#include "cl_platform.h"

#define DATAFILE "main.data"

#ifndef DATASIZE
#define DATASIZE 4 * 1024 * 1024
#endif

#ifndef LOCALSIZE
#define LOCALSIZE 8
#endif

#define NUMGROUPS DATASIZE / LOCALSIZE

// opencl objects:
cl_platform_id Platform;
cl_device_id Device;
cl_kernel Kernel;
cl_program Program;
cl_context Context;
cl_command_queue CmdQueue;

float hX[DATASIZE];
float hY[DATASIZE];

float hSumx4[DATASIZE];
float hSumx3[DATASIZE];
float hSumx2[DATASIZE];
float hSumx[DATASIZE];
float hSumxy[DATASIZE];
float hSumx2y[DATASIZE];
float hSumy[DATASIZE];

const char* CL_FILE_NAME = { "main.cl" };

// function prototypes:
void SelectOpenclDevice();
char* Vendor(cl_uint);
char* Type(cl_device_type);
void Wait(cl_command_queue);
float Determinant(float[3], float[3], float[3]);
void Solve(float[3][3], float[3], float[3]);
void Solve3(float, float, float, float, float, float, float, int, float*, float*, float*);

int main(int argc, char* argv[])
{
    // see if we can even open the opencl kernel program
    // (no point going on if we can't):

    FILE* fp;
#ifdef WIN32
    errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
    if (err != 0)
#else
    fp = fopen(CL_FILE_NAME, "r");
    if (fp == NULL)
#endif
    {
        fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
        return 1;
    }

    cl_int status; // returned status from opencl calls -- test against CL_SUCCESS

    // get the platform id and the device id:

    SelectOpenclDevice(); // sets the global variables Platform and Device

    // 2. create the host memory buffers:

    // read the data file:

    FILE* fdata;
#ifdef WIN32
    err = fopen_s(&fdata, DATAFILE, "r");
    if (err != 0)
#else
    fdata = fopen(DATAFILE, "r");
    if (fdata == NULL)
#endif
    {
        fprintf(stderr, "Cannot open data file '%s'\n", DATAFILE);
        return -1;
    }

    float x, y;
    for (int i = 0; i < DATASIZE; i++) {
        fscanf(fdata, "%f %f", &x, &y);
        hX[i] = x;
        hY[i] = y;
    }
    fclose(fdata);

    // 3. create an opencl context:

    Context = clCreateContext(NULL, 1, &Device, NULL, NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateContext failed\n");

    // 4. create an opencl command queue:

    CmdQueue = clCreateCommandQueue(Context, Device, 0, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateCommandQueue failed\n");

    // 5. allocate the device memory buffers:

    size_t xySize = DATASIZE * sizeof(float);

    cl_mem dX = clCreateBuffer(Context, CL_MEM_READ_ONLY, xySize, NULL, &status);
    cl_mem dY = clCreateBuffer(Context, CL_MEM_READ_ONLY, xySize, NULL, &status);
    cl_mem dSumx4 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx3 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx2 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx2y = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumxy = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumy = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);

    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateBuffer failed\n");

    // 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

    status = clEnqueueWriteBuffer(CmdQueue, dX, CL_FALSE, 0, xySize, hX, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");
    status = clEnqueueWriteBuffer(CmdQueue, dY, CL_FALSE, 0, xySize, hY, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

    Wait(CmdQueue);

    // 7. read the kernel code from a file ...

    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* clProgramText = new char[fileSize + 1]; // leave room for '\0'
    size_t n = fread(clProgramText, 1, fileSize, fp);
    clProgramText[fileSize] = '\0';
    fclose(fp);
    if (n != fileSize)
        fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n);

    // ... and create the kernel program:

    char* strings[1];
    strings[0] = clProgramText;
    Program = clCreateProgramWithSource(Context, 1, (const char**)strings, NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateProgramWithSource failed\n");
    delete[] clProgramText;

    // 8. compile and link the kernel code:

    char* options = { (char*)"" };
    status = clBuildProgram(Program, 1, &Device, options, NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t size;
        clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        cl_char* log = new cl_char[size];
        clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
        fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
        delete[] log;
    }

    // 9. create the kernel object:

    Kernel = clCreateKernel(Program, "Regression", &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateKernel failed\n");

    // 10. setup the arguments to the kernel object:

    status = clSetKernelArg(Kernel, 0, sizeof(cl_mem), &dX);
    status = clSetKernelArg(Kernel, 1, sizeof(cl_mem), &dY);
    status = clSetKernelArg(Kernel, 2, sizeof(cl_mem), &dSumx4);
    status = clSetKernelArg(Kernel, 3, sizeof(cl_mem), &dSumx3);
    status = clSetKernelArg(Kernel, 4, sizeof(cl_mem), &dSumx2);
    status = clSetKernelArg(Kernel, 5, sizeof(cl_mem), &dSumx);
    status = clSetKernelArg(Kernel, 6, sizeof(cl_mem), &dSumx2y);
    status = clSetKernelArg(Kernel, 7, sizeof(cl_mem), &dSumxy);
    status = clSetKernelArg(Kernel, 8, sizeof(cl_mem), &dSumy);

    // 11. enqueue the kernel object for execution:

    size_t globalWorkSize[3] = { DATASIZE, 1, 1 };
    size_t localWorkSize[3] = { LOCALSIZE, 1, 1 };

    Wait(CmdQueue);

    double time0 = omp_get_wtime();

    status = clEnqueueNDRangeKernel(CmdQueue, Kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

    Wait(CmdQueue);
    double time1 = omp_get_wtime();

    // 12. read the results buffer back from the device to the host:

    status = clEnqueueReadBuffer(CmdQueue, dSumx4, CL_FALSE, 0, xySize, hSumx4, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumx3, CL_FALSE, 0, xySize, hSumx3, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumx2, CL_FALSE, 0, xySize, hSumx2, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumx, CL_FALSE, 0, xySize, hSumx, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumx2y, CL_FALSE, 0, xySize, hSumx2y, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumxy, CL_FALSE, 0, xySize, hSumxy, 0, NULL, NULL);
    status = clEnqueueReadBuffer(CmdQueue, dSumy, CL_FALSE, 0, xySize, hSumy, 0, NULL, NULL);
    Wait(CmdQueue);
    float sumx = 0.;
    float sumx4 = 0.;
    float sumx3 = 0.;
    float sumx2 = 0.;
    float sumx2y = 0.;
    float sumxy = 0.;
    float sumy = 0.;
    for (int i = 0; i < DATASIZE; i++) {
        sumx += hSumx[i];
        sumx2 += hSumx2[i];
        sumx3 += hSumx3[i];
        sumx4 += hSumx4[i];
        sumy += hSumy[i];
        sumxy += hSumxy[i];
        sumx2y += hSumx2y[i];
    }
    float Q, L, C;
    Solve3(sumx4, sumx3, sumx2, sumx, sumx2y, sumxy, sumy, DATASIZE, &Q, &L, &C);

#define CSV
#ifdef CSV
    fprintf(stderr, "%8d , %6d , %10.2lf\n",
        DATASIZE, LOCALSIZE, (double)DATASIZE / (time1 - time0) / 1000000.);
#else
    fprintf(stderr, "Array Size: %8d , Work Elements: %4d , MegaPointsProcessedPerSecond: %10.2lf, (%7.1f,%7.1f,%7.1f)\n",
        DATASIZE, LOCALSIZE, (double)DATASIZE / (time1 - time0) / 1000000., Q, L, C);
#endif

    // 13. clean everything up:

    clReleaseKernel(Kernel);
    clReleaseProgram(Program);
    clReleaseCommandQueue(CmdQueue);
    clReleaseMemObject(dSumx2);
    clReleaseMemObject(dSumx);
    clReleaseMemObject(dSumxy);
    clReleaseMemObject(dSumy);

    return 0;
}

// wait until all queued tasks have taken place:

void Wait(cl_command_queue queue)
{
    cl_event wait;
    cl_int status;

    status = clEnqueueMarker(queue, &wait);
    if (status != CL_SUCCESS)
        fprintf(stderr, "Wait: clEnqueueMarker failed\n");

    status = clWaitForEvents(1, &wait);
    if (status != CL_SUCCESS)
        fprintf(stderr, "Wait: clWaitForEvents failed\n");
}

// vendor ids:
#define ID_AMD 0x1002
#define ID_INTEL 0x8086
#define ID_NVIDIA 0x10de

void SelectOpenclDevice()
{
    // select which opencl device to use:
    // priority order:
    //	1. a gpu
    //	2. an nvidia or amd gpu
    //	3. an intel gpu
    //	4. an intel cpu

    int bestPlatform = -1;
    int bestDevice = -1;
    cl_device_type bestDeviceType;
    cl_uint bestDeviceVendor;
    cl_int status; // returned status from opencl calls
                   // test against CL_SUCCESS

    // find out how many platforms are attached here and get their ids:

    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (1)\n");

    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (2)\n");

    for (int p = 0; p < (int)numPlatforms; p++) {
        // find out how many devices are attached to each platform and get their ids:

        cl_uint numDevices;

        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        cl_device_id* devices = new cl_device_id[numDevices];
        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        for (int d = 0; d < (int)numDevices; d++) {
            cl_device_type type;
            cl_uint vendor;
            size_t sizes[3] = { 0, 0, 0 };

            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);

            // select:

            if (bestPlatform < 0) // not yet holding anything -- we'll accept anything
            {
                bestPlatform = p;
                bestDevice = d;
                Platform = platforms[bestPlatform];
                Device = devices[bestDevice];
                bestDeviceType = type;
                bestDeviceVendor = vendor;
            } else // holding something already -- can we do better?
            {
                if (bestDeviceType == CL_DEVICE_TYPE_CPU) // holding a cpu already -- switch to a gpu if possible
                {
                    if (type == CL_DEVICE_TYPE_GPU) // found a gpu
                    { // switch to the gpu we just found
                        bestPlatform = p;
                        bestDevice = d;
                        Platform = platforms[bestPlatform];
                        Device = devices[bestDevice];
                        bestDeviceType = type;
                        bestDeviceVendor = vendor;
                    }
                } else // holding a gpu -- is a better gpu available?
                {
                    if (bestDeviceVendor == ID_INTEL) // currently holding an intel gpu
                    { // we are assuming we just found a bigger, badder nvidia or amd gpu
                        bestPlatform = p;
                        bestDevice = d;
                        Platform = platforms[bestPlatform];
                        Device = devices[bestDevice];
                        bestDeviceType = type;
                        bestDeviceVendor = vendor;
                    }
                }
            }
        }
        delete[] devices;
    }
    delete[] platforms;

    if (bestPlatform < 0) {
        fprintf(stderr, "I found no OpenCL devices!\n");
        exit(1);
    }
    // fprintf(stderr, "I have selected Platform #%d, Device #%d: ", bestPlatform, bestDevice);
    // fprintf(stderr, "Vendor = %s, Type = %s\n", Vendor(bestDeviceVendor), Type(bestDeviceType));
}

char* Vendor(cl_uint v)
{
    switch (v) {
    case ID_AMD:
        return (char*)"AMD";
    case ID_INTEL:
        return (char*)"Intel";
    case ID_NVIDIA:
        return (char*)"NVIDIA";
    }
    return (char*)"Unknown";
}

char* Type(cl_device_type t)
{
    switch (t) {
    case CL_DEVICE_TYPE_CPU:
        return (char*)"CL_DEVICE_TYPE_CPU";
    case CL_DEVICE_TYPE_GPU:
        return (char*)"CL_DEVICE_TYPE_GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return (char*)"CL_DEVICE_TYPE_ACCELERATOR";
    }
    return (char*)"Unknown";
}

float Determinant(float c0[3], float c1[3], float c2[3])
{
    float d00 = c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]);
    float d01 = c1[0] * (c0[1] * c2[2] - c0[2] * c2[1]);
    float d02 = c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
    return d00 - d01 + d02;
}

void Solve(float A[3][3], float X[3], float B[3])
{
    float col0[3] = { A[0][0], A[1][0], A[2][0] };
    float col1[3] = { A[0][1], A[1][1], A[2][1] };
    float col2[3] = { A[0][2], A[1][2], A[2][2] };
    float d0 = Determinant(col0, col1, col2);
    float dq = Determinant(B, col1, col2);
    float dl = Determinant(col0, B, col2);
    float dc = Determinant(col0, col1, B);

    float q = dq / d0;
    float l = dl / d0;
    float c = dc / d0;
    X[0] = q;
    X[1] = l;
    X[2] = c;
}

void Solve3(float sumx4, float sumx3, float sumx2, float sumx, float sumx2y, float sumxy, float sumy, int datasize, float* Q, float* L, float* C)
{
    float A[3][3];
    A[0][0] = sumx4;
    A[0][1] = sumx3;
    A[0][2] = sumx2;
    A[1][0] = sumx3;
    A[1][1] = sumx2;
    A[1][2] = sumx;
    A[2][0] = sumx2;
    A[2][1] = sumx;
    A[2][2] = (float)DATASIZE;

    float Y[3] = { sumx2y, sumxy, sumy };

    float X[3];

    Solve(A, X, Y);

    *Q = X[0];
    *L = X[1];
    *C = X[2];
}
