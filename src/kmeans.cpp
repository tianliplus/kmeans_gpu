#include "kmeans.h"
#include "argparse.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int main(int argc, char **argv) {
	struct options_t opts;
	get_opts(argc, argv, &opts);

	// Read and parse file
	std::ifstream infile(opts.inputfilename);
	std::string item;
	infile >> item;
	int total_points = std::stoi(item);
	double *points[total_points];
	for (int i  = 0; i < total_points; i++) {
		points[i] = (double*)malloc(sizeof(double) * opts.dims);
		// discard point id
		infile >> item;
		for (int j = 0; j < opts.dims; j++) {
			infile >> item;
			points[i][j] = std::stod(item);
		}
	}

	// Init centers
	double *centers[opts.num_cluster];
	kmeans_srand(opts.seed);
	for (int i = 0; i < opts.num_cluster; i++){
	    int index = kmeans_rand() % total_points;
	    centers[i] = (double*)malloc(sizeof(double) * opts.dims);
		for (int j = 0; j < opts.dims; j++) {
			centers[i][j] = points[index][j];
			std::cout << centers[i][j] << " ";
		}
		std::cout << std::endl;
	}

	// Cluster label of the point is the idx of the cluster center
	int labels[total_points];

	int iter_to_converge = 0;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	if (strcmp(opts.implement_type, "cpu") == 0) {
		iter_to_converge = kmeans_cpu(points, centers, labels, opts.num_cluster, opts.dims, total_points, opts.max_num_iter, opts.threshold, opts.unchanged_converge);
	} else {
		std::cout << "Unsupported implement_type: " << opts.implement_type << std::endl;
		exit(0);
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double time_per_iter_in_ms = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 / iter_to_converge;
	printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);

	if (opts.centroids) {
		for (int clusterId = 0; clusterId < opts.num_cluster; clusterId++) {
			printf("%d ", clusterId);
			for (int d = 0; d < opts.dims; d++)
				printf("%lf ", centers[clusterId][d]);
			printf("\n");
		}
	} else {
		printf("clusters:");
		for (int p = 0; p < total_points; p++)
		    printf(" %d", labels[p]);
	}
	for (int i = 0; i < total_points; i++) {
		free(points[i]);
	}
	for (int i = 0; i < opts.num_cluster; i++) {
		free(centers[i]);
	}
}

int kmeans_cpu(double **points, double **centers, int *labels, int k, int dims, int total_points, int max_num_iter, double threshold, bool unchanged_converge) {
	int iter = 0;
	bool done = false;

	while (!done) {
		bool cluster_unchanged = true;
		iter++;
		double *new_centers[k];
		int tmp_space = sizeof(double) * dims;
		for (int i = 0; i < k; i++) {
			new_centers[i] = (double*)malloc(tmp_space);
			memset(new_centers[i], 0, tmp_space);
		}

		int cluster_points_count[k];
		memset(cluster_points_count, 0, k * sizeof(int));
		for (int i = 0; i < total_points; i++) {
			int point_label = 0;
			double distance = calc_distance(points[i], centers[0], dims);
			for (int j = 1; j < k; j++) {
				double tmp_distance = calc_distance(points[i], centers[j], dims);
				if (distance > tmp_distance) {
					point_label = j;
					distance = tmp_distance;
				}
			}
			if (labels[i] != point_label) {
				labels[i] = point_label;
				cluster_unchanged = false;
			}

			for (int j = 0; j < dims; j++) {
				new_centers[point_label][j] += points[i][j];
			}
			cluster_points_count[point_label]++;
		}

		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dims; j++) {
				new_centers[i][j] /= cluster_points_count[i];
			}
		}

		if (iter > max_num_iter || (unchanged_converge && cluster_unchanged)) {
			std::cout << "iter: " << iter << ", unchanged: " << (cluster_unchanged ? "true" : "false") << std::endl;
			done = true;
		} else {
			done = true;
			for (int i = 0; i < k; i++) {
				double distance_delta = sqrt(calc_distance(centers[i], new_centers[i], dims));
				if (distance_delta > threshold) {
					done = false;
					break;
				} else {
					std::cout << "iter: " << iter << ", center: " << i << ", distance_delta: " << distance_delta << std::endl;
				}
			}
		}

		for (int i = 0; i < k; i++) {
			free(centers[i]);
			centers[i] = new_centers[i];
		}
	}

	return iter;
}

double calc_distance(double *p1, double *p2, int dims) {
	double distance = 0;
	for (int i = 0; i < dims; i++) {
		distance += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	}
	return distance;
}
