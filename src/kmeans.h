#pragma once

__device__ __host__ double calc_distance(double *p1, double *p2, int dims);

int kmeans_cpu(double *points, double *centers, int *labels, int k, int dims, int total_points, int max_num_iter, double threshold, bool unchanged_converge);

int kmeans_cuda(double *points, double *centers, int *labels, int dims, int total_points, int num_clusters, int max_num_iter, double threshold);

__global__ void recluster(double *points, double *centers, int *labels, double *global_new_centers_sum, int *global_cluster_points_cnt, int num_clusters, int dims, int total_points);

__global__ void converge_check(double *centers, double *new_centers_sum, int *cluster_points_cnt, int num_clusters, int dims);
