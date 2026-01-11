#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include "../headers/params.hpp"

#define THREAD_BLOCK_SIZE_1D 256
#define THREAD_BLOCK_SIZE_3D 8

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


__global__ void voxelization(Point* d_input, Voxel* d_output, int num_points) {
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
    
    atomicAdd(&d_output[voxel_idx].num_points, 1); 
}

__global__ void extract_active_voxels(Voxel* d_voxels, Voxel* d_active_voxels, int* d_num_active_voxels) { // TODO warp divergence
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_TOT_VOXELS) return;

    int curr_voxel_final_num_points = d_voxels[idx].num_points;
    if (curr_voxel_final_num_points > MIN_POINTS_IN_VOXEL_TO_RENDER) {
        int out_idx = atomicAdd(d_num_active_voxels, 1);
        d_active_voxels[out_idx] = d_voxels[idx];
    }
}

__global__ void setup_voxels(Voxel* voxels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NUM_VOXELS_X ||
        y >= NUM_VOXELS_Y ||
        z >= NUM_VOXELS_Z)
        return;

    int idx = z * (NUM_VOXELS_X * NUM_VOXELS_Y)
            + y * NUM_VOXELS_X
            + x;

    voxels[idx].x = x;
    voxels[idx].y = y;
    voxels[idx].z = z;
    voxels[idx].num_points = 0;
}


__global__ void reset_voxels(Voxel* voxels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_TOT_VOXELS) return;

    voxels[idx].num_points = 0;
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
    addr.sin_port = htons(WORKER_PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket");
        exit(1);
    }

    listen(server_fd, 1);
    printf("Server listening on port %d...\n", WORKER_PORT);

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
    int num_points = 0, total_received = 0, bytes_expected = 0;

    Voxel* d_voxels_output;
    Voxel* d_active_voxels;
    int*   d_num_active_voxels;
    CHECK(cudaMalloc(&d_voxels_output, NUM_TOT_VOXELS * sizeof(Voxel)));
    CHECK(cudaMalloc(&d_active_voxels, NUM_TOT_VOXELS * sizeof(Voxel)));
    CHECK(cudaMalloc(&d_num_active_voxels, sizeof(int)));

    Voxel* h_active_voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel));
    int    h_num_active_voxels;

    dim3 blockSetupVoxels(THREAD_BLOCK_SIZE_3D, THREAD_BLOCK_SIZE_3D, THREAD_BLOCK_SIZE_3D);
    dim3 gridSetupVoxels((NUM_VOXELS_X + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D,
                         (NUM_VOXELS_Y + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D,
                         (NUM_VOXELS_Z + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D);
    setup_voxels<<<gridSetupVoxels, blockSetupVoxels>>>(d_voxels_output);


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
        
        dim3 blockResetVoxels(THREAD_BLOCK_SIZE_1D);
        dim3 gridResetVoxels((NUM_TOT_VOXELS + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        reset_voxels<<<gridResetVoxels, blockResetVoxels>>>(d_voxels_output); 

        // LANCIO KERNEL voxelization
        dim3 blockVox(THREAD_BLOCK_SIZE_1D);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        voxelization <<<gridVox, blockVox>>>(d_input, d_voxels_output, num_points);
        
        // LANCIO KERNEL active_voxels
        CHECK(cudaMemset(d_num_active_voxels, 0, sizeof(int)));

        dim3 blockActiveVoxel(THREAD_BLOCK_SIZE_1D);
        dim3 gridActiveVoxel((NUM_TOT_VOXELS + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);

        extract_active_voxels<<<gridActiveVoxel, blockActiveVoxel>>>(d_voxels_output,d_active_voxels,d_num_active_voxels);


        // COPIA D2H risultati
        CHECK(cudaMemcpy(&h_num_active_voxels, d_num_active_voxels, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_active_voxels, d_active_voxels, NUM_TOT_VOXELS * sizeof(Voxel), cudaMemcpyDeviceToHost));


        // -----------------------SEND TO RENDERER----------------------------
        int total_sent = 0;
        int bytes_to_send = h_num_active_voxels * sizeof(Voxel);

        if (send(renderer_fd, &h_num_active_voxels, sizeof(int), 0) < 0) {
            perror("Error sending active_count");
            break;
        }

        // 4. Ciclo di invio
        while (total_sent < bytes_to_send) {
            // Nota: usiamo 'sock' e il puntatore specifico passato nella struct
            int sent = send(renderer_fd, (char*)h_active_voxels + total_sent, bytes_to_send - total_sent, 0);

            if (sent < 0) {
                perror("Error sending voxel data inside callback");
                break;
            }
            total_sent += sent;
        }

        printf("Completato invio voxels. Totale: %d bytes.\n", total_sent);


        //cleanUP
        CHECK(cudaFree(d_input));
        free(curr_points);
        
    }

    CHECK(cudaFree(d_voxels_output));
    CHECK(cudaFree(d_active_voxels));
    CHECK(cudaFree(d_num_active_voxels));
    free(h_active_voxels);
    
    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;
}