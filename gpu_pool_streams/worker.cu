// N streams

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
#include "../headers/params.hpp"

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
    cudaStream_t streams[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }

    // events to manage the buffers
    cudaEvent_t h2d_done_events[NUM_BUFFERS];
    cudaEvent_t output_was_sent_events[NUM_BUFFERS];

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

        // creazione eventi
        CHECK(cudaEventCreateWithFlags(&h2d_done_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&output_was_sent_events[i], cudaEventDisableTiming));

        // Inizializzazione eventi per il primo giro
        CHECK(cudaEventRecord(h2d_done_events[i], streams[i]));
        CHECK(cudaEventRecord(output_was_sent_events[i], streams[i]));

    }


    int num_points;
    int i = 0, current_stream = 0, total_received = 0, bytes_expected = 0, next_frame_to_send = 0;
    int buffer_output_ready[NUM_BUFFERS] = {0};

    // -----------------LOOP RICEZIONE----------------------------
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) { 
        
        current_stream = i % NUM_BUFFERS;
        printf("Ricevuti %d punti da elaborare.\n", num_points);
        

        if (i >= NUM_BUFFERS) {
            if (cudaEventQuery(h2d_done_events[current_stream]) != cudaSuccess) {
                CHECK(cudaEventSynchronize(h2d_done_events[current_stream]));
            }
        }

        total_received = 0;
        bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected ) {
            int received = recv(client_fd, (char*)h_pinned_inputs[current_stream] + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

        // ---------------------- VOXELIZATION ----------------------------


        CHECK(cudaMemcpyAsync(d_inputs[current_stream], h_pinned_inputs[current_stream], num_points * sizeof(Point), cudaMemcpyHostToDevice, streams[current_stream]));
        cudaEventRecord(h2d_done_events[current_stream], streams[current_stream]);
        
        CHECK(cudaMemsetAsync(d_voxels_output[current_stream], 0, NUM_TOT_VOXELS * sizeof(int), streams[current_stream]));

        dim3 blockVox(THREAD_BLOCK_SIZE);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);
        voxelization <<<gridVox, blockVox, 0, streams[current_stream]>>>(d_inputs[current_stream], d_voxels_output[current_stream], num_points);
        
        CHECK(cudaStreamWaitEvent(streams[current_stream], output_was_sent_events[current_stream], 0));
        CHECK(cudaMemcpyAsync(h_voxels_output[current_stream], d_voxels_output[current_stream], NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost, streams[current_stream]));



        for (int i = 0; i < NUM_BUFFERS; i++) {
            if (!buffer_output_ready[i]) {
                if (cudaEventQuery(output_was_sent_events[i]) == cudaSuccess) {
                    buffer_output_ready[i] = 1; // La GPU ha finito questo slot
                }
            }
        }

        // --- 3. IL CICLO DI INVIO ORDINATO ---
        // Qui avviene la magia: inviamo SOLO se il frame pronto è quello che ci aspettiamo
        int output_slot = next_frame_to_send % NUM_BUFFERS;
        
        // Continuiamo a inviare finché abbiamo una catena di frame pronti consecutivi
        while (buffer_output_ready[output_slot]) {
            
            // Invio dei dati al renderer
            int total_sent = 0;
            int bytes_to_send = NUM_TOT_VOXELS * sizeof(int);

            // 4. Ciclo di invio
            while (total_sent < bytes_to_send) {
                
                int sent = send(renderer_fd, h_voxels_output[output_slot] + total_sent, bytes_to_send - total_sent, 0);

                if (sent < 0) {
                    perror("Error sending voxel data inside callback");
                    break;
                }
                total_sent += sent;
            }

            printf("Completato invio buffer %d. Totale: %zu bytes.\n", output_slot, total_sent);
            cudaEventRecord(output_was_sent_events[output_slot], streams[(next_frame_to_send+1) % NUM_BUFFERS]); // registro l'evento nello stream successivo per evitare conflitti
            
            // Resettiamo lo stato per il prossimo giro
            buffer_output_ready[output_slot] = 0; 
            
            // Passiamo al prossimo frame atteso
            next_frame_to_send++;
            output_slot = next_frame_to_send % NUM_BUFFERS;
        }

        i++;

    }
    
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }


    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaFreeHost(h_pinned_inputs[i]));
        CHECK(cudaFreeHost(h_voxels_output[i]));
        CHECK(cudaFree(d_inputs[i]));
        CHECK(cudaFree(d_voxels_output[i]));

        // FIX: Distruzione eventi
        CHECK(cudaEventDestroy(h2d_done_events[i]));
        CHECK(cudaEventDestroy(output_was_sent_events[i]));

    }   

    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;

}