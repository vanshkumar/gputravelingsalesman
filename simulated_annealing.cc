#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <curand.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>

#include "metropolis_cuda.cuh"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t start_time;
cudaEvent_t stop_time;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start_time));       \
      gpuErrChk(cudaEventCreate(&stop_time));        \
      gpuErrChk(cudaEventRecord(start_time));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop_time));                     \
      gpuErrChk(cudaEventSynchronize(stop_time));                \
      gpuErrChk(cudaEventElapsedTime(&name, start_time, stop_time));  \
      gpuErrChk(cudaEventDestroy(start_time));                   \
      gpuErrChk(cudaEventDestroy(stop_time));                    \
  }


// Given two vectors of length 2, calculates euclidean distance between them
float euclidean_distance(vector<float> point1, vector<float> point2) {
  float p1 = pow(point1[0] - point2[0], 2.0);
  float p2 = pow(point1[1] - point2[1], 2.0);
  return sqrt(p1 + p2);
}

// Given a list of position vectors of cities, generates a distance matrix
float* generate_distance_matrix(vector<vector<float>> coords) {
  int len = coords.size();
  float *dist_matrix = new float[len*len];

  for (int i = 0; i < len; i++) {
    for (int j = 0; j < len; j++) {
      dist_matrix[i * len + j] = euclidean_distance(coords[i], coords[j]);
    }
  }

  return dist_matrix;
}

// Struct to store information about the city/county data that is read in
struct ProblemData {
  map<int, vector<vector<float>>> locations;
  map<int, vector<string>> names;
  map<int, string> county_names;
  vector<int> county_numbers;
  int num_counties;
};

/* 
  Given an istream that reads from a CSV file with our data, reads in all of
  the data and returns information about the problem
*/ 
ProblemData read_cities(istream& in_stream) {
	int city_idx = 0;
  int header_length = 1;

  ProblemData ret;

  map<int, vector<vector<float>>> locations;
  map<int, vector<string>> names;
  map<int, string> county_names;
  vector<int> county_numbers;

  for (string row; getline(in_stream, row); city_idx++) {
    if (city_idx < header_length) {
      continue;
    }

    // header: x_corr, y_corr, town, county, state, county number

    stringstream str(row);
    string values[6];

    for (int i = 0; i < 6; i++) {
      getline(str, values[i], ',');
    }

    vector<float> location;
    location.push_back(stof(values[0]));
    location.push_back(stof(values[1]));

    const int county_num = stoi(values[5]) - 1;

    if (county_names.find(county_num) != county_names.end()) {
      locations[county_num].push_back(location);
      names[county_num].push_back(values[2]);
    }
    else {
      vector<string> name;
      vector<vector<float>> loc;

      name.push_back(values[2]);
      loc.push_back(location);

      names[county_num] = name;
      locations[county_num] = loc;
      county_names[county_num] = values[3];
      county_numbers.push_back(county_num);
    }
  }

  ret.locations = locations;
  ret.names = names;
  ret.county_names = county_names;
  ret.county_numbers = county_numbers;
  ret.num_counties = (int) county_numbers.size();

  return ret;
}


// Struct to store information about county and city distance matrices
struct DistanceMatrices {
  vector<float *> all_county_matrices;
  vector<int> county_num_cities;
  map<pair<int, int>, pair<int, int>> county_path_matrix;
  float *dist_county_matrix;
};

/*
  Arguments:
    num_counties is how many counties we have

    locations is essentially a dictionary that maps county indices to lists of
    position vectors of whichever cities are in that county

  This function calculates the county and city distance matrices for all of our
  counties.
*/
DistanceMatrices calculate_county_and_city_distance_matrices(int num_counties,
    map<int, vector<vector<float>>> locations) {
  DistanceMatrices ret;

  vector<float *> all_county_matrices;
  vector<int> county_num_cities;
  map<pair<int, int>, pair<int, int>> county_path_matrix;

  for (int county_num1 = 0; county_num1 < num_counties; county_num1++) {
    vector<vector<float>> cities = locations[county_num1];
    all_county_matrices.push_back(generate_distance_matrix(cities));
    county_num_cities.push_back((int) cities.size());

    for (int city_num1 = 0; city_num1 < (int) cities.size(); city_num1++) {
      vector<float> city = cities[city_num1];

      for (int county_num2 = county_num1 + 1; county_num2 < num_counties;
          county_num2++) {
        vector<vector<float>> other_cities = locations[county_num2];

        for (int city_num2 = 0; city_num2 < (int) other_cities.size();
            city_num2++) {
          vector<float> other_city = other_cities[city_num2];

          float distance = euclidean_distance(city, other_city);

          const pair<int, int> indices (county_num1, county_num2);

          int city_index1 = county_path_matrix[indices].first;
          int city_index2 = county_path_matrix[indices].second;

          float prev_dist = euclidean_distance(cities[city_index1],
              other_cities[city_index2]);

          if (distance < prev_dist) {
            const pair<int, int> idx1 (county_num1, county_num2);
            const pair<int, int> val1 (city_num1, city_num2);

            const pair<int, int> idx2 (county_num2, county_num1);
            const pair<int, int> val2 (city_num2, city_num1);

            county_path_matrix[idx1] = val1;
            county_path_matrix[idx2] = val2;
          }
        }
      }
    }
  }

  ret.all_county_matrices = all_county_matrices;
  ret.county_num_cities = county_num_cities;
  ret.county_path_matrix = county_path_matrix;

  float *dist_county_matrix = new float[num_counties * num_counties];

  for (int i = 0; i < num_counties; i++) {
    for (int j = i+1; j < num_counties; j++) {
      const pair<int, int> indices (i, j);

      pair<int, int> two_cities = county_path_matrix[indices];

      vector<float> city1 = locations[i][two_cities.first];
      vector<float> city2 = locations[j][two_cities.second];

      float dist = euclidean_distance(city1, city2);

      dist_county_matrix[i * num_counties + j] = dist;
      dist_county_matrix[j * num_counties + i] = dist;
    }
  }

  ret.dist_county_matrix = dist_county_matrix;

  return ret;
}

// Udpate energy given path, distance matrix, and old energy - calculates
// energy of new hypothetical path where indices between idx1 and idx2 are
// reversed
float update_E(vector<int> path, float *dist_matrix, float E_old,
  int idx1, int idx2, int size) {

  float E_new = E_old;

  if (idx1 != idx2) {
      idx2--;

      int start_city = path[idx1];
      int end_city = path[idx2];

      E_new -= dist_matrix[path[idx1-1] * size + start_city];
      E_new -= dist_matrix[end_city * size + path[idx2+1]];

      E_new += dist_matrix[path[idx1-1] * size + path[idx2]];
      E_new += dist_matrix[path[idx1] * size + path[idx2+1]];
  }

  return E_new;

}

// Calculate cost/energy of a path given distance/cost matrix
float calc_energy(vector<int> path, float *dist_matrix, int size) {
  float E = 0.0;
  for (int i = 0; i < (path.size() - 1); i++) {
    E += dist_matrix[path[i] * size + path[i+1]];
  }

  return E;
}

float pi_ratio(float E_old, float E_new, float temp) {
  return exp(-1.0 / temp * (E_new - E_old));
}


// Initialize path randomly with given start and end city indices
vector<int> initialize_path(int size, int start, int end) {
  vector<int> path;
  for (int i = 0; i < size; i++) {
    path.push_back(i);
  }

  path.erase(remove(path.begin(), path.end(), start), path.end());
  if (start != end) {
    path.erase(remove(path.begin(), path.end(), end), path.end());
  }

  random_shuffle(path.begin(), path.end());

  path.insert(path.begin(), start);
  path.push_back(end);

  return path;
}

// Reverse elements of path between index a and b
void reverse_elements_path(vector<int> &path, int a, int b) {
  reverse(path.begin() + a, path.begin() + b);
}

// This struct contains information about the path returned by the metropolis
// algorithm
struct PathInfo {
  vector<int> path;
  float cost;
};


/*
  Arguments:
    distance_matrix is a float array of size num_cities * num_cities that
    contains the cost/distance numbers between cities

    num_cities is how many cities we are creating a path for

    temp is our current tempereature

    start and end are the indices of the cities we want to start and end our
    tour at

    iterations is the number of iterations we want to run metropolis for

  This algorithm performs the metropolis/simulated annealing algorithm
*/
PathInfo iterative_metropolis(float *distance_matrix, int num_cities, float temp,
    int start, int end, int iterations) {

  PathInfo ret;

  // Set up initial path with appropriate start & end
  vector<int> path = initialize_path(num_cities, start, end);

  // Calculate initial energy
  float energy = calc_energy(path, distance_matrix, num_cities);

  for (int iter = 0; iter < iterations; iter++) {
    // Pick random indicies to reverse part of our path
    int idx1 = 0;
    int idx2 = 0;

    while (idx1 == idx2) {
      idx1 = rand() % (path.size() - 3) + 1;
      idx2 = rand() % (path.size() - 3) + 1;
    }

    int a = min(idx1, idx2);
    int b = max(idx1, idx2);

    // Update energy with hypothetical reversal of path
    float energy_new = update_E(path, distance_matrix, energy,
      a, b, num_cities);

    // Calculate energy ratio and create random float for acceptance check
    float check = min(float(1.0), pi_ratio(energy, energy_new, temp));
    float u = (float) (rand()) / (float) (RAND_MAX);

    if (u < check) {
      // Update temperature, energy, and actually perform reversal on path
      temp = 0.9999 * temp;
      reverse_elements_path(path, a, b);
      energy = energy_new;
    }
  }

  ret.path = path;
  ret.cost = energy;

  return ret;
}


/*
  Arguments:
    blocks is # of blocks for our GPU code

    threadsPerBlock is # of threads per block for GPU code

    distance_matrix is a float array of size num_cities * num_cities that
    contains the cost/distance numbers between cities

    num_cities is how many cities we are creating a path for

    temp is our current tempereature

    start and end are the indices of the cities we want to start and end our
    tour at

    num_sims is how many simulations of the simulated annealing algorithm we
    want to perform in parallel

    iterations is the number of iterations we want to run metropolis for
*/
PathInfo gpu_metropolis(int blocks, int threadsPerBlock,
    float *distance_matrix, int num_cities, float temp,
    int start, int end, int num_sims, int iterations) {

  PathInfo ret;

  // Calculate the length of the actual path (+1 if we start and end at the
  // same city, as that is repeating a city)
  int path_length = num_cities;
  if (start == end) {
    path_length++;
  }

  // Set up timing variables
  // float allocation_time, init_time, iterations_time, energy_computation_time;

  // START_TIMER();

  // Allocate GPU memory for distance matrix and copy matrix from host to GPU
  float *dev_dist_matrix;
  gpuErrChk(cudaMalloc((void **) &dev_dist_matrix, num_cities * num_cities * sizeof(float)));
  gpuErrChk(cudaMemcpy(dev_dist_matrix, distance_matrix,
    num_cities * num_cities * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate GPU memory for PRNG states
  curandState *dev_states;
  gpuErrChk(cudaMalloc((void **) &dev_states, num_sims * sizeof(curandState)));

  // Allocate GPU memory for storing each simulations' energies
  float *dev_energies;
  gpuErrChk(cudaMalloc((void **) &dev_energies, num_sims * sizeof(float)));

  // Allocate GPU memory for storing each simulations' current temperatures
  float *dev_temps;
  gpuErrChk(cudaMalloc((void **) &dev_temps, num_sims * sizeof(float)));

  // Allocate GPU 2D array for storing each simulations' path
  int *dev_path;
  gpuErrChk(cudaMalloc((void **) &dev_path, path_length * num_sims * sizeof(int)));


  // STOP_RECORD_TIMER(allocation_time);

  // printf("time for allocation: %f\n", allocation_time);

  // START_TIMER();

  // Initialize path, temperatures, energies, and PRNG states
  cudaCallInitializeDataKernel(blocks, threadsPerBlock, dev_path,
    dev_dist_matrix, dev_states, dev_energies, dev_temps,
    temp, start, end, num_cities, path_length, num_sims);

  // STOP_RECORD_TIMER(init_time);

  // printf("time for initialization: %f\n", init_time);

  // START_TIMER();

  // float kernel_time;

  // START_TIMER();

  // cudaCallParallelMetropolisKernel(blocks, threadsPerBlock, dev_path,
  //     dev_dist_matrix, dev_states, dev_energies, dev_temps,
  //     num_cities, path_length, num_sims,);

  // STOP_RECORD_TIMER(kernel_time);

  // printf("kernel time: %f s\n", kernel_time / 1000.0);

  // Perform the parallelized simulated annealing algorithm
  for (int iter = 0; iter < iterations; iter++) {
    cudaCallParallelMetropolisKernel(blocks, threadsPerBlock, dev_path,
      dev_dist_matrix, dev_states, dev_energies, dev_temps,
      num_cities, path_length, num_sims);
  }

  // STOP_RECORD_TIMER(iterations_time);

  // printf("time for iterations: %f\n", iterations_time);


  // START_TIMER();

  // Copy energy values over from GPU to CPU to find minimum
  float *host_energies = new float[num_sims];
  cudaMemcpy(host_energies, dev_energies, num_sims * sizeof(float),
    cudaMemcpyDeviceToHost);

  // Use thrust to find position of smallest cost of our simulations
  float *min_cost_position = thrust::min_element(host_energies,
    host_energies + num_sims);

  ret.cost = *min_cost_position;

  // Use thrust distance to find which # simulation has lowest cost
  int location = thrust::distance(host_energies, min_cost_position);

  // Get the path with the lowest cost and return it
  int *best_path = new int[path_length];
  cudaMemcpy(best_path, dev_path + location * path_length,
    path_length * sizeof(int), cudaMemcpyDeviceToHost);
  vector<int> final_path (best_path, best_path + path_length);

  ret.path = final_path;

  // STOP_RECORD_TIMER(energy_computation_time);

  // printf("time for energy computation: %f\n", energy_computation_time);

  // float total_time = allocation_time + init_time + iterations_time + energy_computation_time;

  // printf("total time for gpu metropolis algorithm: %f s\n", total_time / 1000.0);

  // printf("\n");

  // Free all the memory we used
  delete best_path;
  delete host_energies;

  cudaFree(dev_dist_matrix);
  cudaFree(dev_states);
  cudaFree(dev_energies);
  cudaFree(dev_path);
  cudaFree(dev_temps);

  return ret;
}

int main(int argc, char** argv) {
  if(argc == 7) {
    // cudaSetDevice(0);

    // First argument is data CSV file
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();

    // Constants related to parallelization
    int blocks = atoi(argv[2]);
    int threadsPerBlock = atoi(argv[3]);

    // Problem parameters
    int num_iters = atoi(argv[4]);
    int num_simulations = atoi(argv[5]);

    // File we will write results to
    ofstream ofs(argv[6]);

    // Read in information from the CSV
    ProblemData data = read_cities(buffer);

    // Calculate county and city distance matrices given problem data
    DistanceMatrices matrices = calculate_county_and_city_distance_matrices(
        data.num_counties, data.locations);

    // Seed random # generator
    srand(time(NULL));

    // This is the index of Los Angeles county
    int start_county = 18;
    int end_county = 18;

    // Initial temperature of 50
    float temp_i = 50.0;

    // County matrix and number of counties are used often, so just declare
    float *dist_county_matrix = matrices.dist_county_matrix;
    int num_counties = data.num_counties;



    // Iterative version
    float iterative_time;

    printf("START iterative version. \n\n");

    START_TIMER();

    // Run iterative simulated annealing on higher-level county problem
    PathInfo host_county_result = iterative_metropolis(dist_county_matrix,
        num_counties, temp_i, start_county, end_county, num_iters);

    // Determine which city index we want to start at (starting from Pasadena)
    int pasadena_idx = 0;
    int start_city = 0;
    for (int i = 0; i < matrices.county_num_cities[start_county]; i++) {
      string name = data.names[start_county][i];
      if (name.find("Pasadena") != string::npos) {
        start_city = i;
        pasadena_idx = i;
        break;
      }
    }

    // Find paths for cities within each county now
    vector<int> host_county_path = host_county_result.path;
    float total_iter_cost = host_county_result.cost;

    for (int i = 0; i < (int(host_county_path.size()) - 1); i++) {
      // We are starting from start_city in county1 and going to end_city in
      // county1. 
      int county1 = host_county_path[i];
      int county2 = host_county_path[i+1];

      // The first element of this pair is the city in county1 we want to end
      // up at after our tour of county1 and the second element is the city in
      // county2 that we want to start our tour of county2 at
      const pair<int, int> idx (county1, county2);
      pair<int, int> two_cities = matrices.county_path_matrix[idx];

      int end_city = two_cities.first;

      float *dist_city_matrix = matrices.all_county_matrices[county1];

      // Calculate best tour given start and end cities
      PathInfo host_city_result = iterative_metropolis(dist_city_matrix,
        matrices.county_num_cities[county1], 2 * temp_i, start_city,
        end_city, num_iters);

      // Start city in next county (county2)
      start_city = two_cities.second;

      total_iter_cost += host_city_result.cost;
    }

    printf("total iterative cost: %f\n", total_iter_cost);

    STOP_RECORD_TIMER(iterative_time);

    printf("iterative time: %f s \n\n", iterative_time / 1000.0);





    printf("START GPU version. \n\n");

    float gpu_time;

    START_TIMER();

    // Run GPU simulated annealing on higher-level county problem 
    PathInfo gpu_county_result = gpu_metropolis(blocks, threadsPerBlock,
        dist_county_matrix, num_counties, temp_i, start_county, end_county,
        num_simulations, num_iters);

    // Start at pasadena still
    start_city = pasadena_idx;

    // Find paths for cities within each county now
    vector<int> gpu_county_path = gpu_county_result.path;
    float total_gpu_cost = gpu_county_result.cost;

    // Lists of floats that contain x and y positions in the order of our
    // overall tour throughout California
    vector<float> overall_x;
    vector<float> overall_y;

    for (int i = 0; i < (int(gpu_county_path.size()) - 1); i++) {
      // The logic here is the same as in the iterative version of this
      // section of the code
      int county1 = gpu_county_path[i];
      int county2 = gpu_county_path[i+1];

      const pair<int, int> idx (county1, county2);
      pair<int, int> two_cities = matrices.county_path_matrix[idx];

      int end_city = two_cities.first;

      float *dist_city_matrix = matrices.all_county_matrices[county1];

      // Calculate best tour given start and end cities
      PathInfo gpu_city_result = gpu_metropolis(blocks, threadsPerBlock,
        dist_city_matrix, matrices.county_num_cities[county1], 2 * temp_i,
        start_city, end_city, num_simulations, num_iters);

      // Here, we add all of the cities in the tour we just found to our
      // overall tour
      vector<int> city_path = gpu_city_result.path;
      for (int i = 0; i < int(city_path.size()); i++) {
        vector<float> position = data.locations[county1][city_path[i]];

        overall_x.push_back(position[0]);
        overall_y.push_back(position[1]);
      }

      start_city = two_cities.second;

      // Add our start_city in county2 to our overall tour
      vector<float> transition_position = data.locations[county2][start_city];
      overall_x.push_back(transition_position[0]);
      overall_y.push_back(transition_position[1]);

      total_gpu_cost += gpu_city_result.cost;
    }

    // Write overall x and overall y (our path) to a file for visualization
    for (int i = 0; i < (overall_x.size() - 1); i++) {
      float x = overall_x[i];
      float y = overall_y[i];

      ofs << x << " " << y << endl;
    }
    ofs << overall_x[overall_x.size() - 1] << " " <<
      overall_y[overall_y.size() - 1];


    printf("total gpu cost: %f\n", total_gpu_cost);

    STOP_RECORD_TIMER(gpu_time);

    printf("total gpu time: %f s \n\n", gpu_time / 1000.0);

    printf("The speedup over the iterative algorithm is %f\n",
      num_simulations * iterative_time / gpu_time);

    printf("The score improvement is %f\n",
      (total_iter_cost - total_gpu_cost) / total_gpu_cost);

    ofs.close();

  }
  else {
    printf("Usage <data csv file> <threadsPerBlock> <num_blocks> <num_iters> <num_simulations> <output_data file> \n");
  }
}
