#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "metropolis_cuda.cuh"


/*
    Raandomly shuffle the path
*/
__device__ void shuffle(int *path, curandState localState,
    int path_length) {

    for (int i = 1; i < (path_length - 1); i++) {
        int j = int(curand_uniform(&localState) * float(path_length - 3.0) + 1.0);
        int temp = path[j];
        path[j] = path[i];
        path[i] = temp;
    }
}


/*
    Calculate the energy of a given path with the given distance/cost matrix
*/
__device__ float calculate_energy(int *path, float *dist_matrix,
    int num_cities, int path_length) {

	float E = 0.0;
	for (int i = 0; i < (path_length - 1); i++) {
		E += dist_matrix[path[i] * num_cities + path[i+1]];
	}

	return E;
}


/* 
    Given a path, distance/cost matrix, the old energy of this path, the
    two indices between which all cities will be reversed, the number of
    cities, and the path_length, compute the new energy if we were to perform
    the reversal.
*/
__device__ float update_energy(int *path, float *dist_matrix,
    float E_old, int idx1, int idx2, int num_cities,
    int path_length) {

    float E_new = E_old;

    if (idx1 != idx2) {
        int start_city = path[idx1];
        int end_city = path[idx2];

        E_new -= dist_matrix[path[idx1-1] * num_cities + start_city];
        E_new -= dist_matrix[end_city * num_cities + path[idx2+1]];

        E_new += dist_matrix[path[idx1-1] * num_cities + path[idx2]];
        E_new += dist_matrix[path[idx1] * num_cities + path[idx2+1]];
    }

    return E_new;
}


__device__ float calc_pi_ratio(float E_old, float E_new, float temp) {
	return exp(-1.0 / temp * (E_new - E_old));
}


/*
    Arguments:
        path is an integer array of size num_simulations * num_cities, that
        contains the paths for each of the simulations we are running

        dist_matrix is a float array of size num_cities * num_cities, that
        contains all of the distance information between cities

        states is a curandState array of size num_simulations that contains the
        states of all of the PRNGs

        energies is a float array of size num_simulations that contains the
        energy/cost values for all of our simulations

        temperatures is a float array of size num_simulations that contains the
        current temperature for each simulation

        num_cities is how many cities we have

        path_length is the length of our path

        num_simulations is how many simulations we are running

    This kernel performs one iteration of the simulated annealing algorithm for
    each of our simulations.
*/
__global__ void
cudaParallelMetropolisKernel(int *path, float *dist_matrix,
        curandState *states, float *energies, float *temperatures,
        int num_cities, int path_length, int num_simulations) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < num_simulations) {
        float E_old = energies[index];
		float temp = temperatures[index];
    
        int *path_new = path + index * path_length;

		// Figure out which two indices in the path are going to be reversed
        int idx1 = int(curand_uniform(&states[index]) * float(path_length - 3.0) + 1.0);
        int idx2 = int(curand_uniform(&states[index]) * float(path_length - 3.0) + 1.0);
        int a = min(idx1, idx2);
        int b = max(idx1, idx2);
        

		// Calculate energy of new path with hypothetical reversal
		float E_new = update_energy(path_new, dist_matrix, E_old,
            a, b, num_cities, path_length);

        // Calculate energy ratio of old & new path
		float check = min(float(1.0), calc_pi_ratio(E_old, E_new, temp));
        
		// Generate random number to see if we accept
		float u = curand_uniform(&states[index]);

		if (u < check) {
            // If accept, change temperature & energy & actually change path
			temperatures[index] = 0.9999 * temp;
			energies[index] = E_new;

            // Reverse portion of path
            int t;
            while (a < b) {
                t = path_new[a];
                path_new[a] = path_new[b];
                path_new[b] = t;

                a++;
                b--;
            }
		}


		index += blockDim.x * gridDim.x;
	}
}


/*
    Given a start and end index, initializes a random path
*/
__device__ void initializePath(int *path, int start, int end,
    curandState localState, int path_length,
    int num_simulations) {

    // Initialize path to just sequence of cities with given start & end
    path[0] = start;
    int actual_counter = 1;
    for (int i = 0; i < path_length; i++) {
        if (i != start && i != end) {
            path[actual_counter] = i;
            actual_counter++;
        }
    }
    path[path_length - 1] = end;

    // Perform shuffles of path to initialize a random path
    for (int i = 0; i < 50; i++) {
        shuffle(path, localState, path_length);
    }
}


// Initializes random initial paths, energies, temperatures, and curand states
__global__ void cudaInitializeDataKernel(int *path, float *dist_matrix,
        curandState *states, float *energies,
        float *temperatures, float init_temp,
        int start, int end, int num_cities, 
		int path_length, int num_simulations) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < num_simulations) {
		// Initialize curand state for this index
		curand_init(index, 0, 0, &states[index]);

		// Initialize random path
		initializePath(path + index * path_length, start, end,
            states[index], path_length, num_simulations);

		// Intiialize energy associated with path
        energies[index] = calculate_energy(path + index * path_length,
            dist_matrix, num_cities, path_length);

		// Initialize annealing temperature
		temperatures[index] = init_temp;

		index += blockDim.x * gridDim.x;
	}
}


void cudaCallParallelMetropolisKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *path, float *dist_matrix,
        curandState *states,
        float *energies, float *temperatures,
        int num_cities, int path_length,
        int num_simulations) {

	cudaParallelMetropolisKernel<<<blocks, threadsPerBlock>>> (path,
		dist_matrix, states, energies, temperatures,
        num_cities, path_length, num_simulations);
}


void cudaCallInitializeDataKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *path, float *dist_matrix,
        curandState *states, float *energies,
        float *temperatures, float init_temp,
        int start, int end, int num_cities,
        int path_length, int num_simulations) {

    cudaInitializeDataKernel<<<blocks, threadsPerBlock>>> (path,
        dist_matrix, states, energies, temperatures,
        init_temp, start, end, num_cities, path_length,
        num_simulations);
}