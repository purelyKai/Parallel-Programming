#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Random number generator seed
unsigned int seed = 0;

// Function prototypes
float Ranf(float, float);
float SQR(float);
void InitBarrier(int);
void WaitBarrier();
void Deer();
void Grain();
void Watcher();
void Weeds();

// Global variables for barrier
omp_lock_t Lock;
volatile int NumInThreadTeam;
volatile int NumAtBarrier;
volatile int NumGone;

// Global variables for simulation state
int NowYear; // 2025 - 2030
int NowMonth; // 0 - 11
float NowPrecip; // inches of rain per month
float NowTemp; // temperature this month
float NowHeight; // grain height in inches
int NowNumDeer; // number of deer in the current population
float NowWeedDensity; // density of weeds (0 to 1)

// Constants
const float GRAIN_GROWS_PER_MONTH = 12.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0; // average
const float AMP_PRECIP_PER_MONTH = 6.0; // plus or minus
const float RANDOM_PRECIP = 2.0; // plus or minus noise

const float AVG_TEMP = 60.0; // average
const float AMP_TEMP = 20.0; // plus or minus
const float RANDOM_TEMP = 10.0; // plus or minus noise

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

const float WEED_GROWTH_RATE = 0.1; // growth rate per month
const float MAX_WEED_DENSITY = 1.0; // maximum density
const float WEED_IMPACT_ON_GRAIN = 0.5; // how much weeds reduce grain growth (0-1)
const float WEED_DEATH_WINTER = 0.7; // how much weeds die off in winter

// Random number generator function
float Ranf(float low, float high)
{
    float r = (float)rand(); // 0 - RAND_MAX
    float t = r / (float)RAND_MAX; // 0. - 1.

    return low + t * (high - low);
}

// Squaring function
float SQR(float x)
{
    return x * x;
}

// Barrier functions
void InitBarrier(int n)
{
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

void WaitBarrier()
{
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if (NumAtBarrier == NumInThreadTeam) {
            NumGone = 0;
            NumAtBarrier = 0;
            // let all other threads get back to what they were doing
            // before this one unlocks, knowing that they might immediately
            // call WaitBarrier() again:
            while (NumGone != NumInThreadTeam - 1)
                ;
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);

    while (NumAtBarrier != 0)
        ; // this waits for the nth thread to arrive

#pragma omp atomic
    NumGone++; // this flags how many threads have returned
}

// Deer thread function
void Deer()
{
    while (NowYear < 2031) {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)(NowHeight);
        if (nextNumDeer < carryingCapacity)
            nextNumDeer++;
        else if (nextNumDeer > carryingCapacity)
            nextNumDeer--;

        if (nextNumDeer < 0)
            nextNumDeer = 0;

        // DoneComputing barrier:
        WaitBarrier();

        // copy the value into the global variable:
        NowNumDeer = nextNumDeer;

        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();
    }
}

// Grain thread function
void Grain()
{
    while (NowYear < 2031) {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.));
        float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.));

        float nextHeight = NowHeight;
        // Adjust grain growth based on temperature, precipitation, and weed density
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH * (1.0 - WEED_IMPACT_ON_GRAIN * NowWeedDensity);
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

        if (nextHeight < 0.)
            nextHeight = 0.;

        // DoneComputing barrier:
        WaitBarrier();

        // copy the value into the global variable:
        NowHeight = nextHeight;

        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();
    }
}

// Watcher thread function
void Watcher()
{
    // Vector to store data for later display
    std::vector<std::string> data;

    // Add header to data
    data.push_back("Month,Year,Temp,Precip,Height,Deer,WeedDensity");

    while (NowYear < 2031) {
        // DoneComputing barrier:
        WaitBarrier();

        // DoneAssigning barrier:
        WaitBarrier();

        // print the current state variables:
        char buffer[256];
        sprintf(buffer, "%d,%d,%.2f,%.2f,%.2f,%d,%.2f",
            NowMonth + 1, NowYear, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowWeedDensity);
        data.push_back(buffer);

        // Calculate the next temperature and precipitation:
        NowMonth++;
        if (NowMonth >= 12) {
            NowMonth = 0;
            NowYear++;
        }

        float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.); // angle of earth around the sun
        float temp = AVG_TEMP - AMP_TEMP * cos(ang);
        NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
        NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
        if (NowPrecip < 0.)
            NowPrecip = 0.;

        // DonePrinting barrier:
        WaitBarrier();
    }

    // Print all collected data at the end
    for (const std::string& line : data) {
        std::cout << line << std::endl;
    }
}

// Weeds thread function
void Weeds()
{
    while (NowYear < 2031) {
        // Compute next weed density based on current conditions
        float nextWeedDensity = NowWeedDensity;

        // Weed growth depends on temperature and precipitation
        // Weeds grow faster in warmer, wetter conditions
        float tempFactor = exp(-SQR((NowTemp - 70.0) / 30.)); // Weeds prefer warmer temps than grain
        float precipFactor = exp(-SQR((NowPrecip - 8.0) / 10.)); // Weeds prefer slightly less water than ideal for grain

        // Winter die-off (when temperature drops significantly)
        bool isWinter = (NowMonth == 11 || NowMonth == 0 || NowMonth == 1);
        float seasonalFactor = isWinter ? (1.0 - WEED_DEATH_WINTER) : 1.0;

        // Update weed density - affected by current conditions and grain height
        // Weeds grow less if grain is tall (shading effect)
        float growthRate = WEED_GROWTH_RATE * tempFactor * precipFactor * (1.0 - NowHeight / 20.0);
        nextWeedDensity += growthRate;
        nextWeedDensity *= seasonalFactor;

        // Ensure weed density is between 0 and MAX_WEED_DENSITY
        if (nextWeedDensity > MAX_WEED_DENSITY)
            nextWeedDensity = MAX_WEED_DENSITY;
        if (nextWeedDensity < 0.0)
            nextWeedDensity = 0.0;

        // DoneComputing barrier:
        WaitBarrier();

        // copy the value into the global variable:
        NowWeedDensity = nextWeedDensity;

        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();
    }
}

int main(int argc, char* argv[])
{
    // Starting date and time:
    NowMonth = 0;
    NowYear = 2025;

    // Starting state:
    NowNumDeer = 2;
    NowHeight = 5.;
    NowWeedDensity = 0.1; // Initial weed density

    // Create initial temperature and precipitation
    float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.); // angle of earth around the sun
    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
    if (NowPrecip < 0.)
        NowPrecip = 0.;

    // Initialize random number generator
    srand(seed);

    // Set up the barrier
    omp_set_num_threads(4); // Number of threads to use
    InitBarrier(4);

// Setup and launch parallel sections
#pragma omp parallel sections
    {
#pragma omp section
        {
            Deer();
        }

#pragma omp section
        {
            Grain();
        }

#pragma omp section
        {
            Watcher();
        }

#pragma omp section
        {
            Weeds();
        }
    } // implied barrier -- all functions must return in order
      // to allow any of them to get past here

    return 0;
}
