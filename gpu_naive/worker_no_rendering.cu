#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>

#include "params.hpp"

#define THREAD_BLOCK_SIZE 8

// Macro per controllo errori CUDA
#define CHECK(call) \
do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while (0)

__global__ void vectorGeneration(float4* d_vectors) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int nx = gridDim.x * blockDim.x;
    int ny = gridDim.y * blockDim.y;
    int idx = z * (nx * ny) + y * nx + x;

    if (x >= NUM_VOXELS_X || y >= NUM_VOXELS_Y || z >= NUM_VOXELS_Z) return;

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

    int curr_voxel_x = (int)floor((point.x - MIN_X) / DIM_VOXEL);
    int curr_voxel_y = (int)floor((point.y - MIN_Y) / DIM_VOXEL);
    int curr_voxel_z = (int)floor((point.z - MIN_Z) / DIM_VOXEL);
    
    if(curr_voxel_x < 0 || curr_voxel_x >= NUM_VOXELS_X ||
       curr_voxel_y < 0 || curr_voxel_y >= NUM_VOXELS_Y ||
       curr_voxel_z < 0 || curr_voxel_z >= NUM_VOXELS_Z) {
            return;
    }

    int voxel_idx = curr_voxel_z * (NUM_VOXELS_X* NUM_VOXELS_Y) + curr_voxel_y * NUM_VOXELS_X + curr_voxel_x;
    atomicAdd(&d_output[voxel_idx], 1); 
}

int main(void) {
    // 1. FIX BUFFERING: Disabilita il buffer di output per vedere i log subito
    setbuf(stdout, NULL); 
    printf("Avvio Worker CUDA...\n");

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
    
    // SETUP SOCKET
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("Error creating socket"); exit(1); }

    // Opzionale: Permette di riusare la porta subito se il programma crasha e lo riavvii
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket"); exit(1);
    }

    listen(server_fd, 1);
    printf("Server listening on port %d... Waiting for Supplier...\n", PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) { perror("Error accepting request"); exit(1); }
    
    printf("Connected with supplier!\n");

    int num_points;
    Point* d_input;
    int* d_output;
    int* voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
    
    // LOOP RICEZIONE
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) { 
        
        printf("Ricevuti %d punti da elaborare.\n", num_points);
        curr_points = (Point*) malloc(num_points * sizeof(Point));
        
        // 2. FIX RICEZIONE: Ricevi tutto il blocco in una volta
        int total_received = 0;
        int bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected) {
            int received = recv(client_fd, (char*)curr_points + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

        // --- VOXELIZATION ---
        CHECK(cudaMalloc(&d_input, num_points * sizeof(Point)));
        CHECK(cudaMemcpy(d_input, curr_points, num_points * sizeof(Point), cudaMemcpyHostToDevice)); 
        
        CHECK(cudaMalloc(&d_output, NUM_TOT_VOXELS * sizeof(int)));
        CHECK(cudaMemset(d_output, 0, NUM_TOT_VOXELS * sizeof(int))); 

        dim3 blockVox(THREAD_BLOCK_SIZE);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);
        
        voxelization <<<gridVox, blockVox>>>(d_input, d_output, num_points);

        CHECK(cudaMemcpy(voxels, d_output, NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost));
        printf("Voxelization completata per questo frame.\n");

        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        free(curr_points);
    }
    
    printf("Supplier disconnesso. Chiusura Worker.\n");
    free(voxels);
    free(vectorTranslations);
    close(client_fd);
    close(server_fd);

    return 0;
}