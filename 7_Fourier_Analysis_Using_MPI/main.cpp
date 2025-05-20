#define _USE_MATH_DEFINES
#include <math.h>
#include <mpi.h>
#include <stdio.h>

#define F_2_PI ((float)(2. * M_PI))

// which node is in charge?

#define THEBOSS 0

// files to read and write:

#define BIGSIGNALFILEBIN (char*)"bigsignal.bin"
#define BIGSIGNALFILEASCII (char*)"bigsignal.txt"
#define CSVPLOTFILE (char*)"plot.csv"

// tag to "scatter":

#define TAG_SCATTER 'S'

// tag to "gather":

#define TAG_GATHER 'G'

// how many elements are in the big signal:

#define NUMELEMENTS (1 * 1024 * 1024)

// only consider this many periods (this is enough to uncover the secret sine waves):

#define MAXPERIODS 100

// which file type to read, BINARY or ASCII (BINARY is much faster to read):

// #define ASCII
#define BINARY

// print debugging messages?

#define DEBUG true

// globals:

int NumCpus; // total # of cpus involved
float* BigSums; // the overall MAXPERIODS autocorrelation array
float* BigSignal; // the overall NUMELEMENTS-big signal data
int PPSize; // per-processor local array size
float* PPSums; // per-processor autocorrelation sums
float* PPSignal; // per-processor local array to hold the sub-signal

// function prototype:

void DoOneLocalFourier(int);

int main(int argc, char* argv[])
{
    MPI_Status status;

    MPI_Init(&argc, &argv);

    int me; // which one I am

    MPI_Comm_size(MPI_COMM_WORLD, &NumCpus);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    // decide how much data to send to each processor:

    PPSize = NUMELEMENTS / NumCpus; // assuming it comes out evenly

    // local arrays:

    PPSignal = new float[PPSize]; // per-processor signal
    PPSums = new float[MAXPERIODS]; // per-processor sums of the products

    // read the BigSignal array:

    if (me == THEBOSS) // this is the big-data-owner
    {
        BigSignal = new float[NUMELEMENTS]; // to hold the entire signal

#ifdef ASCII
        FILE* fp = fopen(BIGSIGNALFILEASCII, "r");
        if (fp == NULL) {
            fprintf(stderr, "Cannot open signal file '%s'\n", BIGSIGNALFILEASCII);
            return -1;
        }

        for (int i = 0; i < NUMELEMENTS; i++) {
            float f;
            fscanf(fp, "%f", &f);
            BigSignal[i] = f;
        }
#endif

#ifdef BINARY
        FILE* fp = fopen(BIGSIGNALFILEBIN, "rb");
        if (fp == NULL) {
            fprintf(stderr, "Cannot open signal file '%s'\n", BIGSIGNALFILEBIN);
            return -1;
        }

        fread(BigSignal, sizeof(float), NUMELEMENTS, fp);
#endif
    }

    // create the array to hold all the sums:

    if (me == THEBOSS) {
        BigSums = new float[MAXPERIODS]; // where all the sums will go
    }

    // start the timer:

    double time0 = MPI_Wtime();

    // have THEBOSS send to itself (really not a "send", just a copy):

    if (me == THEBOSS) {
        for (int i = 0; i < PPSize; i++) {
            PPSignal[i] = BigSignal[THEBOSS * PPSize + i];
        }
    }

    // getting the signal data distributed:

    if (me == THEBOSS) {
        // have the THEBOSS send to everyone else:
        for (int dst = 0; dst < NumCpus; dst++) {
            if (dst != THEBOSS)
                MPI_Send(&BigSignal[dst * PPSize], PPSize, MPI_FLOAT, dst, TAG_SCATTER, MPI_COMM_WORLD);
        }
    } else {
        // have everyone else receive from the THEBOSS:
        MPI_Recv(PPSignal, PPSize, MPI_FLOAT, THEBOSS, TAG_SCATTER, MPI_COMM_WORLD, &status);
    }

    // each processor does its own fourier:

    DoOneLocalFourier(me);

    // get the sums back:

    if (me == THEBOSS) {
        // get THEBOSS's sums:
        for (int p = 0; p < MAXPERIODS; p++) {
            BigSums[p] = PPSums[p]; // start the overall sums with the THEBOSS's sums
        }
    } else {
        // each processor sends its sums back to the THEBOSS:
        MPI_Send(PPSums, MAXPERIODS, MPI_FLOAT, THEBOSS, TAG_GATHER, MPI_COMM_WORLD);
    }

    // THEBOSS receives the sums and adds them into the overall sums:

    if (me == THEBOSS) {
        float tmpSums[MAXPERIODS];
        for (int src = 0; src < NumCpus; src++) {
            if (src != THEBOSS) {
                MPI_Recv(tmpSums, MAXPERIODS, MPI_FLOAT, src, TAG_GATHER, MPI_COMM_WORLD, &status);

                for (int p = 0; p < MAXPERIODS; p++)
                    BigSums[p] += tmpSums[p];
            }
        }
    }

    // stop the timer:

    double time1 = MPI_Wtime();

    // print the performance:

#define CSV
    if (me == THEBOSS) {
        double seconds = time1 - time0;
        double performance = (double)NumCpus * (double)MAXPERIODS * (double)PPSize / seconds / 1000000.; // mega-sums per second
#ifdef CSV
        fprintf(stderr, "%3d , %10d , %9.2lf\n", NumCpus, NUMELEMENTS, performance);
#else
        fprintf(stderr, "%3d processors, %10d elements, %9.2lf mega-sums per second\n",
            NumCpus, NUMELEMENTS, performance);
#endif
    }

    // write the file to be plotted to look for the secret periods:

    if (me == THEBOSS) {
        FILE* fp = fopen(CSVPLOTFILE, "w");
        if (fp == NULL) {
            fprintf(stderr, "Cannot write to plot file '%s'\n", CSVPLOTFILE);
        } else {
            for (int p = 0; p < MAXPERIODS; p++) {
                fprintf(fp, "%6d , %10.2f\n", p, BigSums[p]);
            }
            fclose(fp);
        }
    }

    // all done:

    MPI_Finalize();
    return 0;
}

// read from the per-processor signal array, write into the local sums array:

void DoOneLocalFourier(int me)
{
    if (DEBUG)
        fprintf(stderr, "Node %3d entered DoOneLocalFourier( )\n", me);

    PPSums[0] = 0.;
    for (int p = 1; p < MAXPERIODS; p++) {
        PPSums[p] = 0.;
        float omega = F_2_PI / (float)p; // frequency in radians/element
        for (int t = 0; t < PPSize; t++) {
            int element = me * PPSize + t;
            PPSums[p] += PPSignal[t] * sinf(omega * (float)element);
        }
    }
}
