// 3 streams:
//  - 1 H2D
//  - 1 Kernel
//  - 1 D2H

// uso memoria host pinned per trasfermenti + malloc iniziale

// uso di cuda events per sincronizzare gli stream tra loro

// allocazione del numero massimo di punti (stimato se fosse un caso reale) per ospitare i punti ogni tot su device

// più buffer per ospitare i punti sul device, non un solo buffer con offset (è più complesso da gestire e non cambia praticamente niente)
// uso di cuda events per segnalare quando un buffer è stato elaborato e quindi può essere sovrascritto (un evento che segnala che il buffer è libero)
// uso di ringbuffer per gestire l'uso dei buffer

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
#define NUM_BUFFERS 2
#define THREAD_BLOCK_SIZE 8
#define MAX_POINTS_PER_BUFFER 131100

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
    // -------------------------- SETUP SOCKET COMMUNICATION --------------------
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points;
    Point* d_input;

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

    // -------------------------- SOCKET VERSO RENDERER --------------------
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


    // ------------------------CUDA STREAMS SETUP -----------------

    // creating the streams
    cudaStream_t h2d, kernel, d2h;
    CHECK(cudaStreamCreate(&h2d));
    CHECK(cudaStreamCreate(&kernel));
    CHECK(cudaStreamCreate(&d2h));

    // creating NUM_BUFFERS buffers to manage multiple frames a time as inputs
    Point* h_pinned_inputs[NUM_BUFFERS];
    Point* d_inputs[NUM_BUFFERS];

    // creating NUM_BUFFERS buffers for output voxels
    int* d_voxels_output[NUM_BUFFERS];
    int* h_voxels_output[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        // alloco memoria host pinned sulla ram per i frame in input
        CHECK(cudaMallocHost((void**)&h_pinned_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));
        // alloco memoria device per input
        CHECK(cudaMalloc((void**)&d_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));

        // alloco memoria host pinned per i voxels in output
        CHECK(cudaMallocHost((void**)&h_voxels_output[i], NUM_TOT_VOXELS * sizeof(int)));
        // alloco memoria device per output
        CHECK(cudaMalloc((void**)&d_voxels_output[i], NUM_TOT_VOXELS * sizeof(int)));
    }

    cudaEvent_t buffer_input_free_events[NUM_BUFFERS]; // one event for each input buffer to signal that the buffer can be overwritten
    cudaEvent_t h2d_done_event[NUM_BUFFERS]; // one event for each output buffer to signal that the buffer can be overwritten
    cudaEvent_t kernel_done_event[NUM_BUFFERS]; // one event for each buffer to signal that the kernel has computed the informations inside the input buffer
    cudaEvent_t buffer_output_done_events[NUM_BUFFERS];
    
    for (int i = 0; i < NUM_BUFFERS; i++) { 
        CHECK(cudaEventCreate(&buffer_input_free_events[i]));
        CHECK(cudaEventCreate(&buffer_output_done_events[i]));
        CHECK(cudaEventCreate(&h2d_done_event[i]));
        CHECK(cudaEventCreate(&kernel_done_event[i]));
        // inizialmente tutti i buffer sono free
        CHECK(cudaEventRecord(buffer_input_free_events[i], 0));
        CHECK(cudaEventRecord(buffer_output_done_events[i], 0));
    }   

    int num_points;
    int i = 0, curr_buffer_sent = 0, count_buffer_sent = 0;
    int current_buffer = 0;

    // LOOP RICEZIONE
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) { 
        
        current_buffer = i % NUM_BUFFERS;
        printf("Ricevuti %d punti da elaborare.\n", num_points);
        
        // 2. FIX RICEZIONE: Ricevi tutto il blocco in una volta
        int total_received = 0;
        int bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected) {
            int received = recv(client_fd, (char*)h_pinned_inputs[current_buffer] + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

        // --- VOXELIZATION ---
        if (cudaEventQuery(buffer_input_free_events[current_buffer]) != cudaSuccess) {
            // il buffer input non è ancora libero, aspetto
            CHECK(cudaEventSynchronize(buffer_input_free_events[current_buffer]));
        }

        CHECK(cudaMemcpyAsync(d_inputs[current_buffer], h_pinned_inputs[current_buffer], num_points * sizeof(Point), cudaMemcpyHostToDevice, h2d)); 
        CHECK(cudaEventRecord(h2d_done_event[current_buffer], h2d));

        if (cudaEventQuery(buffer_output_done_events[current_buffer]) != cudaSuccess) {
            // il buffer output non è ancora libero, aspetto
            CHECK(cudaEventSynchronize(buffer_output_done_events[current_buffer]));


            // PROBLEMAAAA 1


            // facendo la event synchronize, l'evento buffer output done viene resettato
            // quindi quando sotto vado a rifare la cudaEventRecord non vedo più l'evento precedente e non scrivo niente sulla socket


            // PROBLEMAAAA 2

            
            // non dobbiamo usare la event synchronize che rallenta notevolmente l'inserimento di operazioni asincrone negli stream
            // capire se ricreare un nuovo evento per ogni iterazione del ciclo

        }

        CHECK(cudaMemsetAsync(d_voxels_output[current_buffer], 0, NUM_TOT_VOXELS * sizeof(int), d2h)); 

        dim3 blockVox(THREAD_BLOCK_SIZE);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);
        
        if (cudaEventQuery(h2d_done_event[current_buffer]) != cudaSuccess) {
            // l'host to device non è ancora finito, aspetto
            CHECK(cudaEventSynchronize(h2d_done_event[current_buffer]));
        }

        voxelization <<<gridVox, blockVox, 0, kernel>>>(d_inputs[current_buffer], d_voxels_output[current_buffer], num_points);
        CHECK(cudaEventRecord(kernel_done_event[current_buffer], kernel));
        CHECK(cudaEventRecord(buffer_input_free_events[current_buffer], h2d));

        if (cudaEventQuery(kernel_done_event[current_buffer]) != cudaSuccess) {
            // il kernel non è ancora finito, aspetto
            CHECK(cudaEventSynchronize(kernel_done_event[current_buffer]));
        }

        CHECK(cudaMemcpyAsync(h_voxels_output[current_buffer], d_voxels_output[current_buffer], NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost, d2h));
        CHECK(cudaEventRecord(buffer_output_done_events[current_buffer], d2h));
        
        // Controllo se ci sono dei buffer cpu pronti da inviare e in caso li invio
        curr_buffer_sent = 0;
        for (int j=1; j < NUM_BUFFERS + 1; j++) {
            if (cudaEventQuery(buffer_output_done_events[(count_buffer_sent + j) % NUM_BUFFERS]) == cudaSuccess) {
                

                //NON ENTRA MAI IN QUESTO RAMO
                printf("Invio del buffer %d al renderer...\n", (count_buffer_sent + j) % NUM_BUFFERS);

                int buffer_idx = (count_buffer_sent + j) % NUM_BUFFERS;
                int bytes_to_send = NUM_TOT_VOXELS * sizeof(int);
                int total_sent = 0;
                
                // Ensure all data is sent
                while (total_sent < bytes_to_send) {
                    int sent = send(renderer_fd, (char*)h_voxels_output[buffer_idx] + total_sent, bytes_to_send - total_sent, 0);
                    printf("Inviati %d bytes.\n", sent);

                    if (sent < 0) {
                        perror("Error sending voxel data");
                        break;
                    }
                    total_sent += sent;
                }
                printf("Inviato buffer %d (%d bytes) al renderer.\n", buffer_idx, total_sent);
                curr_buffer_sent++;
            }
            else {
                break;
            }
        }
        count_buffer_sent = (count_buffer_sent + curr_buffer_sent) % NUM_BUFFERS;


        printf("Voxelization completata per questo frame.\n");
        i++;
    }
    

    CHECK(cudaStreamDestroy(h2d));
    CHECK(cudaStreamDestroy(kernel));
    CHECK(cudaStreamDestroy(d2h));

    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaFreeHost(h_pinned_inputs[i]));
        CHECK(cudaFreeHost(h_voxels_output[i]));
        CHECK(cudaFree(d_inputs[i]));
        CHECK(cudaFree(d_voxels_output[i]));
    }   

}