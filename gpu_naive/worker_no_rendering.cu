#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>

#include "params.hpp"
#define PORT 53456

#define THREAD_BLOCK_SIZE 8

#define CHECK(call)                                                     \
do {                                                                    \
    const cudaError_t error = call;                                     \
    if (error != cudaSuccess) {                                         \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                   \
        printf("code:%d, reason: %s\n", error,                          \
               cudaGetErrorString(error));                              \
        exit(1);                                                        \
    }                                                                   \
} while (0)


__global__ void vectorGeneration(float4* d_vectors) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int nx = gridDim.x * blockDim.x;
    int ny = gridDim.y * blockDim.y;
    int idx = z * (nx * ny) + y * nx + x;

    if (x >= NUM_VOXELS_X || y >= NUM_VOXELS_Y || z >= NUM_VOXELS_Z)
        return;

    float4 vector = {
        (x * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_X,
        (y * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Y,
        (z * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Z,
        1.0f
    };

    d_vectors[idx] = vector;

}

__global__ void voxelization(Point* d_input, int* d_output, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point point = d_input[idx];

    // voxelize this point
    int curr_voxel_x = (int)floor((point.x - MIN_X) / DIM_VOXEL);
    int curr_voxel_y = (int)floor((point.y - MIN_Y) / DIM_VOXEL);
    int curr_voxel_z = (int)floor((point.z - MIN_Z) / DIM_VOXEL);
    
    if(curr_voxel_x < 0 || curr_voxel_x >= NUM_VOXELS_X ||
        curr_voxel_y < 0 || curr_voxel_y >= NUM_VOXELS_Y ||
        curr_voxel_z < 0 || curr_voxel_z >= NUM_VOXELS_Z) {
            // punto fuori dai limiti
            return;
    }

    // calcolo indice array lineare voxel
    int voxel_idx = curr_voxel_z * (NUM_VOXELS_X* NUM_VOXELS_Y) + curr_voxel_y * NUM_VOXELS_X + curr_voxel_x;
    
    atomicAdd(&d_output[voxel_idx], 1); 
}



int main(void) {
    
    // -------------------------------- SETUP TRANSLATION VECTORS --------------------------------
    float4* vectorTranslations = (float4*) malloc(NUM_TOT_VOXELS * sizeof(float4));
    float4* d_vectors;
    CHECK(cudaMalloc(&d_vectors, NUM_TOT_VOXELS*sizeof(float4)));
    
    dim3 blockSize(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 gridSize(                           
        (NUM_VOXELS_X + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,           
        (NUM_VOXELS_Y + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,          
        (NUM_VOXELS_Z + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE        
    );

    vectorGeneration <<<gridSize, blockSize>>>(d_vectors);
    CHECK(cudaMemcpy(vectorTranslations, d_vectors, NUM_TOT_VOXELS * sizeof(float4), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_vectors));
    
    // --------------------------SETUP SOCKET COMMUNICATION--------------------
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Error creating socket");
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket");
        exit(1);
    }

    listen(server_fd, 1);
    printf("Server listening on port %d...\n", PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) {
        perror("Error accepting request");
        exit(1);
    }
    printf("Connected with supplier.\n\n");
    int i = 0;

    // ------------------------FRAME BY FRAME COMPUTATIONS-----------------
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    Point point;
    int num_points;
    Point* d_input;
    int* d_output;
    int* voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
    memset(voxels, 0, NUM_TOT_VOXELS * sizeof(int));
    int i = 0;

    while ((recv(client_fd, &num_points, sizeof(int), 0)) != 0) { //recv numero di punti
        curr_points = (Point*) malloc(num_points * sizeof(Point));
        //recv di tutti i punti
        for (int num_points_recvd = 0; num_points_recvd < num_points; num_points_recvd++) {
            recv(client_fd, curr_points + num_points_recvd, sizeof(Point), 0);
        }

        // -----------------------VOXELIZATION-------------------------------
        // ALLOCAZIONE PUNTI
        CHECK(cudaMalloc(&d_input, num_points * sizeof(Point)));
        CHECK(cudaMemcpy(d_input, curr_points, num_points * sizeof(Point), cudaMemcpyHostToDevice)); 
        
        // ALLOCAZIONE VOXELS
        CHECK(cudaMalloc(&d_output, NUM_TOT_VOXELS * sizeof(int)));
        CHECK(cudaMemset(d_output, 0, NUM_TOT_VOXELS * sizeof(int))); 

        // LANCIO KERNEL
        dim3 blockSize(THREAD_BLOCK_SIZE);
        dim3 gridSize((num_points + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);
        voxelization <<<gridSize, blockSize>>>(d_input, d_output, num_points);
        //Copia D2H risultati
        CHECK(cudaMemcpy(voxels, d_output, NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost));
        printf("FINITO FRAME %d\n", i);
        i++;

        // reset risorse
        memset(voxels, 0, NUM_TOT_VOXELS * sizeof(int));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        free(curr_points);
    }
    
    free(voxels);
    free(vectorTranslations);
    close(client_fd);
    close(server_fd);

    return 0;
}