#include "kmeans.h"

#include <stdio.h>

__global__ void converge_check(double *centers, double *new_centers_sum, int *cluster_points_cnt, int num_clusters, int dims) {
    extern __shared__ double tmp_delta[];
    int point_idx = threadIdx.x;
    if (point_idx >= num_clusters) {
        return;
    }

    // TODO shmem?
    int cluster_points_count = cluster_points_cnt[point_idx];

    double delta = 0;
    for (int p = point_idx * dims; p < point_idx * (dims + 1); p++) {
        double new_center = new_centers_sum[p] / cluster_points_count;
        delta += (new_center - centers[p]) * (new_center - centers[p]);
        centers[p] = new_center;
    }
    tmp_delta[threadIdx.x] = sqrt(delta);

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tmp_delta[threadIdx.x] += tmp_delta[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // reuse new_centers_sum[0] to store sum delta
        new_centers_sum[0] = tmp_delta[0];
    }
}

__global__ void recluster(double *points, double *centers, int *labels, double *global_new_centers_sum, int *global_cluster_points_cnt, int num_clusters, int dims, int total_points) {
    extern __shared__ char shared_memory[];

    double *local_centers = (double *)shared_memory;
    int *cluster_points_count = (int *)(shared_memory + dims * num_clusters * sizeof(double));
    double *new_centers_sum = (double *)(shared_memory + dims * num_clusters * sizeof(double) + num_clusters * sizeof(int));

    int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_idx >= total_points) {
        return;
    }

    if (threadIdx.x < num_clusters) {
        for (int d = 0; d < dims; d++) {
            int idx = d + threadIdx.x * dims;
            local_centers[idx] = centers[idx];
            new_centers_sum[idx] = 0;
        }
        cluster_points_count[threadIdx.x] = 0;
    }
    __syncthreads();

    int label = 0;
    double distance = calc_distance(&points[point_idx * dims], &local_centers[0], dims);
    for (int k = 1; k < num_clusters; k++) {
        double tmp_distance = calc_distance(&points[point_idx * dims], &local_centers[k * dims], dims);
        if (distance > tmp_distance) {
            label = k;
            distance = tmp_distance;
        }
    }
    if (labels[point_idx] != label) {
        labels[point_idx] = label;
    }

    atomicAdd(&cluster_points_count[label], 1);
    for (int d = 0; d < dims; d++) {
        atomicAdd(&new_centers_sum[label * dims + d], points[point_idx * dims + d]);
    }
    __syncthreads();

    if (threadIdx.x < num_clusters) {
        int k = threadIdx.x;
        if (cluster_points_count[k] > 0) {
            atomicAdd(&global_cluster_points_cnt[k], cluster_points_count[k]);
        }
        for (int d = 0; d < dims; d++) {
            atomicAdd(&global_new_centers_sum[k * dims + d], new_centers_sum[label * dims + d]);
        }
    }
}

__host__ __device__ double calc_distance(double *p1, double *p2, int dims) {
    double distance = 0;
    for (int i = 0; i < dims; i++) {
        distance += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return distance;
}

int kmeans_cuda(double *points, double *centers, int *labels, int dims, int total_points, int num_cluster, int max_num_iter, double threshold) {
    int labels_bytes = sizeof(int) * total_points;
    int points_bytes = sizeof(double) * total_points * dims;
    int centers_bytes = sizeof(double) * num_cluster * dims;
    int cluster_points_count_bytes = sizeof(int) * dims;

    double *d_points, *d_centers, *d_tmp_centers;
    int *d_labels, *d_cluster_points_count;

    cudaMalloc((void **)&d_points, points_bytes);
    cudaMalloc((void **)&d_centers, centers_bytes);
    cudaMalloc((void **)&d_tmp_centers, centers_bytes);
    cudaMalloc((void **)&d_labels, labels_bytes);
    cudaMalloc((void **)&d_cluster_points_count, cluster_points_count_bytes);

    cudaMemcpy(d_points, points, points_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, centers, centers_bytes, cudaMemcpyHostToDevice);

    int block_dim = 64;
    int grid_dim = (total_points + block_dim - 1) / block_dim;
    int shared_mem_bytes = num_cluster * dims * 2 * sizeof(double) + num_cluster * sizeof(int);

    int iter = 0;
    bool done = false;
    while (!done) {
        iter++;
        cudaMemset(d_tmp_centers, 0, centers_bytes);
        cudaMemset(d_cluster_points_count, 0, cluster_points_count_bytes);

        recluster<<<grid_dim, block_dim, shared_mem_bytes>>>(d_points, d_centers, d_labels, d_tmp_centers, d_cluster_points_count, num_clusters, dims, total_points);
        converge_check<<<1, num_clusters, num_clusters>>>(d_centers, d_tmp_centers, d_cluster_points_count, num_clusters, dims);
        if (iter > max_num_iter) {
            done = true;
        } else {
            double delta;
            cudaMemcpy(&delta, d_tmp_centers, sizeof(double), cudaMemcpyDeviceToHost);
            if (delta < threshold) {
                done = true;
            }
            std::cout << "iter: " << iter << ", distance_delta: " << std::setprecision(15) << std::fixed << delta << std::endl;
        }
    }

    cudaMemcpy(labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(centers, d_centers, centers_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_tmp_centers);
    cudaFree(d_labels);
    cudaFree(d_cluster_points_count);
    return iter;
}