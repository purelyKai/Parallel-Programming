#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#ifndef F_PI
#define F_PI (float)M_PI
#endif

// print debugging messages?
#ifndef DEBUG
#define DEBUG false
#endif

// setting the number of threads to use:
// (this a default value -- it can also be set from the outside by your script)
#ifndef NUMT
#define NUMT 2
#endif

// setting the number of trials in the monte carlo simulation:
// (this a default value -- it can also be set from the outside by your script)
#ifndef NUMTRIALS
#define NUMTRIALS 50000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES 30
#endif

// ranges for the random numbers:
const float GMIN = 10.0; // ground distance in meters
const float GMAX = 20.0; // ground distance in meters
const float HMIN = 20.0; // cliff height in meters
const float HMAX = 30.0; // cliff height in meters
const float DMIN = 10.0; // distance to castle in meters
const float DMAX = 20.0; // distance to castle in meters
const float VMIN = 20.0; // intial cnnonball velocity in meters / sec
const float VMAX = 30.0; // intial cnnonball velocity in meters / sec
const float THMIN = 70.0; // cannonball launch angle in degrees
const float THMAX = 80.0; // cannonball launch angle in degrees

const float GRAVITY = -9.8; // acceleraion due to gravity in meters / sec^2
const float TOL = 5.0; // tolerance in cannonball hitting the castle in meters
                       // castle is destroyed if cannonball lands between d-TOL and d+TOL

// function prototypes:
float Ranf(float, float);
int Ranf(int, int);
void TimeOfDaySeed();

// degrees-to-radians:
inline float Radians(float degrees) { return (F_PI / 180.f) * degrees; }

// main program:
int main(int argc, char* argv[])
{
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    TimeOfDaySeed(); // seed the random number generator

    omp_set_num_threads(
        NUMT); // set the number of threads to use in parallelizing the for-loop

    // better to define these here so that the rand() calls don't get into the
    // thread timing:
    float* vs = new float[NUMTRIALS];
    float* ths = new float[NUMTRIALS];
    float* gs = new float[NUMTRIALS];
    float* hs = new float[NUMTRIALS];
    float* ds = new float[NUMTRIALS];

    // fill the random-value arrays:
    for (int n = 0; n < NUMTRIALS; n++) {
        vs[n] = Ranf(VMIN, VMAX);
        ths[n] = Ranf(THMIN, THMAX);
        gs[n] = Ranf(GMIN, GMAX);
        hs[n] = Ranf(HMIN, HMAX);
        ds[n] = Ranf(DMIN, DMAX);
    }

    // get ready to record the maximum performance and the probability:
    double maxPerformance = 0.; // must be declared outside the NUMTRIES loop
    int numHits; // must be declared outside the NUMTRIES loop

    // looking for the maximum performance:
    for (int tries = 0; tries < NUMTRIES; tries++) {
        double time0 = omp_get_wtime();

        numHits = 0;

#pragma omp parallel for default(none) shared(vs, ths, gs, hs, ds, numHits)
        for (int n = 0; n < NUMTRIALS; n++) {
            // randomize everything:
            float v = vs[n];
            float thr = Radians(ths[n]);
            float vx = v * cos(thr);
            float vy = v * sin(thr);
            float g = gs[n];
            float h = hs[n];
            float d = ds[n];

            // see if the ball doesn't even reach the cliff:
            float t = -vy / (0.5 * GRAVITY);
            float x = vx * t;

            if (x <= g) {
                if (DEBUG)
                    fprintf(stderr, "Ball doesn't even reach the cliff\n");
            } else {
                // see if the ball hits the vertical cliff face:
                t = g / vx;
                float y = vy * t + 0.5 * GRAVITY * t * t;

                if (y <= h) {
                    if (DEBUG)
                        fprintf(stderr, "Ball hits the cliff face\n");
                } else {
                    // the ball hits the upper deck:
                    // the time solution for this is a quadratic equation of the form:
                    // At^2 + Bt + C = 0.
                    // where 'A' multiplies time^2
                    //       'B' multiplies time
                    //       'C' is a constant
                    float A = 0.5 * GRAVITY;
                    float B = vy;
                    float C = -h;
                    float disc = B * B - 4.f * A * C; // quadratic formula discriminant

                    // ball doesn't go as high as the upper deck:
                    // this should "never happen" ... :-)
                    if (disc < 0.) {
                        if (DEBUG)
                            fprintf(stderr, "Ball doesn't reach the upper deck.\n");
                        exit(1); // something is wrong...
                    }

                    // successfully hits the ground above the cliff:
                    // get the intersection:
                    float sqrtdisc = sqrtf(disc);
                    float t1 = (-B + sqrtdisc) / (2.f * A); // time to intersect high ground
                    float t2 = (-B - sqrtdisc) / (2.f * A); // time to intersect high ground

                    // only care about the second intersection
                    float tmax = t1;
                    if (t2 > t1)
                        tmax = t2;

                    // how far does the ball land horizontlly from the edge of the cliff?
                    float upperDist = vx * tmax - g;

                    // see if the ball hits the castle:
                    if (fabs(upperDist - d) <= TOL) {
                        if (DEBUG)
                            fprintf(stderr, "Hits the castle at upperDist = %8.3f\n", upperDist);
#pragma omp atomic
                        numHits++;
                    } else {
                        if (DEBUG)
                            fprintf(stderr, "Misses the castle at upperDist = %8.3f\n", upperDist);
                    }
                } // if ball clears the cliff face
            } // if ball gets as far as the cliff face
        } // for(# of  monte carlo trials)

        double time1 = omp_get_wtime();
        double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1000000.;
        if (megaTrialsPerSecond > maxPerformance)
            maxPerformance = megaTrialsPerSecond;
    } // for (# of timing tries)

    float probability = (float)numHits / (float)(NUMTRIALS); // just get for the last run

    // uncomment this if you want to print output to a ready-to-use CSV file:

#define CSV
#ifdef CSV
    fprintf(stderr, "%2d , %8d , %6.2f, %6.2lf\n", NUMT, NUMTRIALS, 100.0 * probability, maxPerformance);
#else
    fprintf(stderr,
        "%2d threads : %8d trials ; probability = %6.2f%% ; megatrials/sec = "
        "%6.2lf\n",
        NUMT, NUMTRIALS, 100. * probability, maxPerformance);
#endif

    return 0;
}

float Ranf(float low, float high)
{
    float r = (float)rand(); // 0 - RAND_MAX
    float t = r / (float)RAND_MAX; // 0. - 1.

    return low + t * (high - low);
}

int Ranf(int ilow, int ihigh)
{
    float low = (float)ilow;
    float high = ceil((float)ihigh);

    return (int)Ranf(low, high);
}

// call this if you want to force your program to use
// a different random number sequence every time you run it:
void TimeOfDaySeed()
{
    time_t now;
    time(&now);

    struct tm n;
    struct tm jan01;
#ifdef WIN32
    localtime_s(&n, &now);
    localtime_s(&jan01, &now);
#else
    n = *localtime(&now);
    jan01 = *localtime(&now);
#endif
    jan01.tm_mon = 0;
    jan01.tm_mday = 1;
    jan01.tm_hour = 0;
    jan01.tm_min = 0;
    jan01.tm_sec = 0;

    double seconds = difftime(now, mktime(&jan01));
    unsigned int seed = (unsigned int)(1000. * seconds); // milliseconds
    srand(seed);
}
