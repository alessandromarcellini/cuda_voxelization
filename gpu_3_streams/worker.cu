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


void CUDART_CB send_socket(cudaStream_t stream, cudaError_t status, void *data) {
    // casting del puntatore void* alla nostra struttura
    struct CallbackOldData *args = (struct CallbackOldData *)data;

    // controllo errori cuda precedenti
    if (status != cudaSuccess) {
        printf("Errore stream CUDA prima della callback: %d\n", status);
        // Liberiamo la memoria allocata per gli argomenti prima di uscire
        free(args); 
        return;
    }

    // estrazione dei dati
    int sock = args->socket_fd;
    char* buffer_to_send = (char*)args->buffer_ptr;
    int buf_id = args->buffer_id;
    int active_count = *(args->active_count);
    size_t bytes_to_send = active_count * sizeof(Voxel);

    printf("Callback avviata. Invio del buffer %d (%zu bytes) al renderer...\n", buf_id, bytes_to_send);

    if (send(sock, &active_count, sizeof(int), 0) < 0) {
        perror("Error sending active_count");
        free(args);
        return;
    }

    size_t total_sent = 0;

    // ciclo di invio
    while (total_sent < bytes_to_send) {
        ssize_t sent = send(sock, buffer_to_send + total_sent, bytes_to_send - total_sent, 0);

        if (sent < 0) {
            perror("Error sending voxel data inside callback");
            break;
        }
        total_sent += sent;
    }

    printf("Completato invio buffer %d. Totale: %zu bytes.\n", buf_id, total_sent);
    free(args);
}

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


__global__ void extract_active_voxels(Voxel* d_voxels, Voxel* d_active_voxels, int* d_num_active_voxels) {
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
    // -------------------------- SETUP SOCKET COMMUNICATION --------------------
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
    Voxel* d_voxels_output[NUM_BUFFERS];

    // creating buffers to host active voxels on GPU
    Voxel* d_active_voxels[NUM_BUFFERS];
    int*   d_num_active_voxels[NUM_BUFFERS];

    // creating buffers to host active voxels on CPU
    Voxel* h_active_voxels[NUM_BUFFERS];
    int    h_num_active_voxels[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        // alloco memoria host pinned sulla ram per i frame in input
        CHECK(cudaMallocHost((void**)&h_pinned_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));
        // alloco memoria device per input
        CHECK(cudaMalloc((void**)&d_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));

        // alloco memoria device per output
        CHECK(cudaMalloc((void**)&d_voxels_output[i], NUM_TOT_VOXELS * sizeof(Voxel)));

        dim3 blockSetupVoxels(THREAD_BLOCK_SIZE_3D, THREAD_BLOCK_SIZE_3D, THREAD_BLOCK_SIZE_3D);
        dim3 gridSetupVoxels(
            (NUM_VOXELS_X + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D,
            (NUM_VOXELS_Y + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D,
            (NUM_VOXELS_Z + THREAD_BLOCK_SIZE_3D - 1) / THREAD_BLOCK_SIZE_3D
        );
        setup_voxels <<< gridSetupVoxels, blockSetupVoxels, 0, 0 >>>(d_voxels_output[i]);

        //alloco memoria per contatori e voxel attivi
        CHECK(cudaMalloc((void**)&d_active_voxels[i], NUM_TOT_VOXELS * sizeof(Voxel)));
        CHECK(cudaMalloc((void**)&d_num_active_voxels[i], sizeof(int)));

        CHECK(cudaMallocHost((void**)&h_active_voxels[i], NUM_TOT_VOXELS * sizeof(Voxel)));
    }



    cudaEvent_t buffer_input_free_events[NUM_BUFFERS];
    cudaEvent_t h2d_done_events[NUM_BUFFERS];          
    cudaEvent_t buffer_output_contains_result_events[NUM_BUFFERS]; 
    cudaEvent_t buffer_output_active_voxels_free_events[NUM_BUFFERS];
    cudaEvent_t buffer_output_voxels_free_events[NUM_BUFFERS];    
    cudaEvent_t buffer_output_was_sent_events[NUM_BUFFERS];


    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaEventCreateWithFlags(&buffer_input_free_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&h2d_done_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&buffer_output_contains_result_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&buffer_output_active_voxels_free_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&buffer_output_voxels_free_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&buffer_output_was_sent_events[i], cudaEventDisableTiming));

        // Inizializzazione eventi per il primo giro
        CHECK(cudaEventRecord(buffer_input_free_events[i], kernel));
        CHECK(cudaEventRecord(buffer_output_active_voxels_free_events[i], d2h));
        CHECK(cudaEventRecord(buffer_output_voxels_free_events[i], d2h));
        CHECK(cudaEventRecord(buffer_output_was_sent_events[i], d2h));
    }



    int num_points;
    int i = 0, current_buffer = 0, total_received = 0, bytes_expected = 0;

    // -----------------LOOP RICEZIONE----------------------------
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) { 
        
        current_buffer = i % NUM_BUFFERS;
        printf("Ricevuti %d punti da elaborare.\n", num_points);
        

        if (i >= NUM_BUFFERS) {
            CHECK(cudaEventSynchronize(h2d_done_events[current_buffer]));
        }

        total_received = 0;
        bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected ) {
            int received = recv(client_fd, (char*)h_pinned_inputs[current_buffer] + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

        // ---------------------- VOXELIZATION ----------------------------
        cudaStreamWaitEvent(h2d, buffer_input_free_events[current_buffer], 0);
        CHECK(cudaMemcpyAsync(d_inputs[current_buffer], h_pinned_inputs[current_buffer], num_points * sizeof(Point), cudaMemcpyHostToDevice, h2d));
        cudaEventRecord(h2d_done_events[current_buffer], h2d);
        
        cudaStreamWaitEvent(kernel, buffer_output_voxels_free_events[current_buffer], 0);

        dim3 blockResetVoxels(THREAD_BLOCK_SIZE_1D);
        dim3 gridResetVoxels((NUM_TOT_VOXELS + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        reset_voxels<<<gridResetVoxels, blockResetVoxels, 0, kernel>>>(d_voxels_output[current_buffer]);

        cudaStreamWaitEvent(kernel, h2d_done_events[current_buffer], 0);
        dim3 blockVox(THREAD_BLOCK_SIZE_1D);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        voxelization <<<gridVox, blockVox, 0, kernel>>>(d_inputs[current_buffer], d_voxels_output[current_buffer], num_points);
        cudaEventRecord(buffer_input_free_events[current_buffer], kernel);
        // -------------------------------------------------------------
        // azzera contatore voxel attivi
        cudaStreamWaitEvent(kernel, buffer_output_active_voxels_free_events[current_buffer], 0);
        CHECK(cudaMemsetAsync(d_num_active_voxels[current_buffer],
                            0,
                            sizeof(int),
                            kernel));

        // kernel di compattazione
        dim3 blockActiveVoxel(THREAD_BLOCK_SIZE_1D);
        dim3 gridActiveVoxel((NUM_TOT_VOXELS + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);

        extract_active_voxels<<<gridActiveVoxel, blockActiveVoxel, 0, kernel>>>(
            d_voxels_output[current_buffer],
            d_active_voxels[current_buffer],
            d_num_active_voxels[current_buffer]
        );
        // lancio evento buffer voxels generici libero su stream kernel
        cudaEventRecord(buffer_output_voxels_free_events[current_buffer], kernel);
        //---------------------------------------------------------------
        
        cudaEventRecord(buffer_output_contains_result_events[current_buffer], kernel);

        cudaStreamWaitEvent(d2h, buffer_output_contains_result_events[current_buffer], 0);
        cudaStreamWaitEvent(d2h, buffer_output_was_sent_events[current_buffer], 0);
        
        //copia a host del numero di voxel attivi
        CHECK(cudaMemcpyAsync(&h_num_active_voxels[current_buffer],
                      d_num_active_voxels[current_buffer],
                      sizeof(int),
                      cudaMemcpyDeviceToHost,
                      d2h));
        //copia a host dei voxel attivi
        CHECK(cudaMemcpyAsync(h_active_voxels[current_buffer],
                            d_active_voxels[current_buffer],
                            NUM_TOT_VOXELS * sizeof(Voxel),
                            cudaMemcpyDeviceToHost,
                            d2h));

        // lancio evento buffer voxels attivi libero su stream d2h
        cudaEventRecord(buffer_output_active_voxels_free_events[current_buffer], d2h);

        // --- INIZIO BLOCCO CALLBACK ---

        // allocazione della struttura dati per passare gli argomenti alla callback
        // usiamo malloc perché la struct deve sopravvivere fino all'esecuzione della callback
        struct CallbackOldData *cb_args = (struct CallbackOldData *)malloc(sizeof(struct CallbackOldData));
        
        // Riempimento dati (Socket, Puntatore al buffer specifico, Dimensione, ID)
        cb_args->socket_fd = renderer_fd;         
        cb_args->buffer_ptr = h_active_voxels[current_buffer];
        cb_args->active_count = &(h_num_active_voxels[current_buffer]);
        cb_args->buffer_id = i;

        CHECK(cudaStreamAddCallback(d2h, send_socket, (void*)cb_args, 0));

        // --- FINE BLOCCO CALLBACK ---
        cudaEventRecord(buffer_output_was_sent_events[current_buffer], d2h);
        i++;
    }
    
    
    CHECK(cudaStreamDestroy(h2d));
    CHECK(cudaStreamDestroy(kernel));
    CHECK(cudaStreamDestroy(d2h));


    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaFreeHost(h_pinned_inputs[i]));
        CHECK(cudaFreeHost(h_active_voxels[i]));
        CHECK(cudaFree(d_inputs[i]));
        CHECK(cudaFree(d_voxels_output[i]));
        CHECK(cudaFree(d_num_active_voxels[i]));
        CHECK(cudaFree(d_active_voxels[i]));

        // FIX: Distruzione eventi
        CHECK(cudaEventDestroy(buffer_input_free_events[i]));
        CHECK(cudaEventDestroy(h2d_done_events[i]));
        CHECK(cudaEventDestroy(buffer_output_contains_result_events[i]));
        CHECK(cudaEventDestroy(buffer_output_active_voxels_free_events[i]));
        CHECK(cudaEventDestroy(buffer_output_voxels_free_events[i]));
        CHECK(cudaEventDestroy(buffer_output_was_sent_events[i]));

    }   

    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;

}