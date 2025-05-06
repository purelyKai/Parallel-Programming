#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string>
#include <time.h>

// setting the number of threads:
#ifndef NUMT
#define NUMT 2
#endif

// setting the number of capitals we want to try:
#ifndef NUMCAPITALS
#define NUMCAPITALS 5
#endif

// maximum iterations to allow looking for convergence:
#define MAXITERATIONS 100

// how many tries to discover the maximum performance:
#define NUMTRIES 30

#define CSV

struct city {
    std::string name;
    float longitude;
    float latitude;
    int capitalnumber;
    float mindistance;
};

#include "UsCities.data"

// setting the number of cities we want to try:
#define NUMCITIES (sizeof(Cities) / sizeof(struct city))

struct capital {
    std::string name;
    float longitude;
    float latitude;
    float longsum;
    float latsum;
    int numsum;
};

struct capital Capitals[NUMCAPITALS];

float Distance(int city, int capital)
{
    float dx = Cities[city].longitude - Capitals[capital].longitude;
    float dy = Cities[city].latitude - Capitals[capital].latitude;
    return sqrtf(dx * dx + dy * dy);
}

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    // fprintf(stderr, "OpenMP is supported -- version = %d\n", _OPENMP);
#else
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    // make sure we have the data correctly:
    // for (int i = 0; i < NUMCITIES; i++) {
    //     fprintf(stderr, "%3d  %8.2f  %8.2f  %s\n", i, Cities[i].longitude, Cities[i].latitude, Cities[i].name.c_str());
    // }

    omp_set_num_threads(NUMT); // set the number of threads to use in parallelizing the for-loop:

    // seed the capitals:
    // (this is just picking initial capital cities at uniform intervals)
    for (int k = 0; k < NUMCAPITALS; k++) {
        int cityIndex = k * (NUMCITIES - 1) / (NUMCAPITALS - 1);
        Capitals[k].longitude = Cities[cityIndex].longitude;
        Capitals[k].latitude = Cities[cityIndex].latitude;
    }

    double time0, time1;
    for (int n = 0; n < MAXITERATIONS; n++) {
        // reset the summations for the capitals:
        for (int k = 0; k < NUMCAPITALS; k++) {
            Capitals[k].longsum = 0.;
            Capitals[k].latsum = 0.;
            Capitals[k].numsum = 0;
        }

        time0 = omp_get_wtime();

#pragma omp parallel for default(none) shared(Capitals, Cities)
        for (int i = 0; i < NUMCITIES; i++) {
            int capitalnumber = -1;
            float mindistance = 1.e+37;

            for (int k = 0; k < NUMCAPITALS; k++) {
                float dist = Distance(i, k);
                if (dist < mindistance) {
                    mindistance = dist;
                    capitalnumber = k;
                    Cities[i].capitalnumber = k;
                }
            }

            int k = Cities[i].capitalnumber;

// this is here for the same reason as the Trapezoid noteset uses it:
#pragma omp critical
            {
                Capitals[k].longsum += Cities[i].longitude;
                Capitals[k].latsum += Cities[i].latitude;
                Capitals[k].numsum++;
            }
        }
        time1 = omp_get_wtime();

        // get the average longitude and latitude for each capital:
        for (int k = 0; k < NUMCAPITALS; k++) {
            Capitals[k].longitude = Capitals[k].longsum / Capitals[k].numsum;
            Capitals[k].latitude = Capitals[k].latsum / Capitals[k].numsum;
        }
    }

    double megaCityCapitalsPerSecond = (double)NUMCITIES * (double)NUMCAPITALS / (time1 - time0) / 1000000.;

    // figure out what actual city is closest to each capital:
    for (int k = 0; k < NUMCAPITALS; k++) {
        int minCity = -1;
        float minDist = 1.e+37;

        for (int i = 0; i < NUMCITIES; i++) {
            float dx = Cities[i].longitude - Capitals[k].longitude;
            float dy = Cities[i].latitude - Capitals[k].latitude;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                minCity = i;
            }
        }

        Capitals[k].name = Cities[minCity].name;
    }

    // print the longitude-latitude of each new capital city:
    // you only need to do this once per some number of NUMCAPITALS -- do it for the 1-thread version:
    if (NUMT == 1) {
        for (int k = 0; k < NUMCAPITALS; k++) {
            fprintf(stdout, "\t%3d:  %8.2f , %8.2f , %s\n", k, Capitals[k].longitude, Capitals[k].latitude, Capitals[k].name.c_str());
        }
    }
#ifdef CSV
    fprintf(stderr, "%2d , %4d , %4d , %8.3lf\n", NUMT, NUMCITIES, NUMCAPITALS, megaCityCapitalsPerSecond);
    if (NUMT == 1) {
        fprintf(stdout, "%2d , %4d , %4d , %8.3lf\n", NUMT, NUMCITIES, NUMCAPITALS, megaCityCapitalsPerSecond);
    }
#else
    fprintf(stderr, "%2d threads : %4d cities ; %4d capitals; megatrials/sec = %8.3lf\n",
        NUMT, NUMCITIES, NUMCAPITALS, megaCityCapitalsPerSecond);
#endif
}
