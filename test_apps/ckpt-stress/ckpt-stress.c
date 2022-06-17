
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


#define PRECISION   0.005


int main(int argc, char *argv[]) {
    int rank, nbProcs, arg;
    double w_time, wait_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
      printf("Usage: ./ckpt-stress <size_mb> <time>\n");
      return 1;
    }

    arg = atoi(argv[1]);
    wait_time = (double)atoi(argv[2]);

    if (arg < 0 || wait_time < 0) {
      printf("Usage: ./ckpt-stress <size_mb> <time>\n");
      return 1;
    }

    w_time = MPI_Wtime();
    //iterate through all the data
    uint64_t num = 1024*1024*arg/sizeof(int);
    int *ckpt = (int *)malloc(num*sizeof(int));
    for(uint64_t i = 0; i < num; i++){
      ckpt[i] = (i*i + 256) % 1000451;
    }

    while (1){
      if (MPI_Wtime() - w_time >= wait_time) break;
    }

    printf("Finalizing...\n");
    fflush(stdout);

    free(ckpt);
    MPI_Finalize();
    return 0;
}
