#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include "params.hpp"

#define RENDERER_PORT 60000
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
    
    // -------------------------- SETUP SOCKET SUPPLIER --------------------
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

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

    // -------------------------- SETUP SOCKET RENDERER --------------------
    int renderer_fd;
    struct sockaddr_in renderer_addr;

    renderer_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (renderer_fd < 0) {
        perror("Error creating renderer socket");
        exit(1);
    }

    renderer_addr.sin_family = AF_INET;
    renderer_addr.sin_port = htons(RENDERER_PORT);
    inet_pton(AF_INET, "127.0.0.1", &renderer_addr.sin_addr);

    if (connect(renderer_fd, (struct sockaddr*)&renderer_addr, sizeof(renderer_addr)) < 0) {
        perror("Error connecting to renderer");
        exit(1);
    }

    printf("Connected to renderer on port %d.\n\n", RENDERER_PORT);

    // ------------------------FRAME BY FRAME COMPUTATIONS-----------------
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    Point* curr_points;
    Point* d_input;
    int num_points = 0, total_received = 0, bytes_expected;
    int* d_output;
    int* voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
    memset(voxels, 0, NUM_TOT_VOXELS * sizeof(int));


    while(recv(client_fd, &num_points, sizeof(int), 0) > 0) {

        printf("Ricevuti %d punti da elaborare.\n", num_points);
        curr_points = (Point*) malloc(num_points * sizeof(Point));

        total_received = 0;
        bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected ) {
            int received = recv(client_fd, (char*)curr_points + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura socket
            total_received += received;
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
        //voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
        CHECK(cudaMemcpy(voxels, d_output, NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost));
        

        // -----------------------SEND TO RENDERER----------------------------
        int total_sent = 0;
        int bytes_to_send = NUM_TOT_VOXELS * sizeof(int);

        // 4. Ciclo di invio
        while (total_sent < bytes_to_send) {
            // Nota: usiamo 'sock' e il puntatore specifico passato nella struct
            int sent = send(renderer_fd, voxels + total_sent, bytes_to_send - total_sent, 0);

            if (sent < 0) {
                perror("Error sending voxel data inside callback");
                break;
            }
            total_sent += sent;
        }

        printf("Completato invio voxels. Totale: %d bytes.\n", total_sent);


        //cleanUP
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        free(curr_points);
        
    }

    
    free(voxels);
    
    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;
}