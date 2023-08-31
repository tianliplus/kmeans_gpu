#pragma once

double calc_distance(double *p1, double *p2, int dims);

int kmeans_cpu(double *points, double *& centers, int *labels, int k, int dims, int total_points, int max_num_iter, double threshold, bool unchanged_converge);

int kmeans_cuda(double *points, double *& centers, int *labels, int dims, int total_points, int num_clusters, int max_num_iter, double threshold);

