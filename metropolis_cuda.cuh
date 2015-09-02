#include <curand_kernel.h>

#ifndef CUDA_METROPOLIS_CUH
#define CUDA_METROPOLIS_CUH


void cudaCallParallelMetropolisKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *path, float *dist_matrix,
        curandState *states,
        float *energies, float *temperatures,
        int num_cities, int path_length,
        int num_simulations);


void cudaCallInitializeDataKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *path, float *dist_matrix,
        curandState *states, float *energies,
        float *temperatures, float init_temp,
        int start, int end, int num_cities,
        int path_length, int num_simulations);

#endif
