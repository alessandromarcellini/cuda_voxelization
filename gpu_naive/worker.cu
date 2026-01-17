#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include "../headers/params.hpp"

#define THREAD_BLOCK_SIZE_1D 512
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


__global__ void voxelization(Point* d_input, int* d_num_points_output, int num_points) {
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
    
    atomicAdd(&d_num_points_output[voxel_idx], 1); 
}


// Funzione device di supporto per calcolare l'offset locale nel warp
__device__ int warpPrefixSum(int val, int& total_warp_sum) {
    int laneId = threadIdx.x % 32;
    int sum = val;
    
    // Somma i valori dei thread precedenti in log2(32) passi

    // WARP INTRINSICS : __shfl_up_sync barriera di sincronizzazione a livello di warp
    // la funzione __shfl guarda direttamente dentro i regsitri privati degli altri thread
    // __shfl_up(mask, val, delta) legge il valore di val del thread con indice diminuito di delta (a sinistra)
    // il sync serve per far eseguire ai thread specificati nella mask, nel nostro caso un warp intero
    // la lettura contemporaneamente

    int n = __shfl_up_sync(0xffffffff, sum, 1);
    if (laneId >= 1) sum += n;
    n = __shfl_up_sync(0xffffffff, sum, 2);
    if (laneId >= 2) sum += n;
    n = __shfl_up_sync(0xffffffff, sum, 4);
    if (laneId >= 4) sum += n;
    n = __shfl_up_sync(0xffffffff, sum, 8);
    if (laneId >= 8) sum += n;
    n = __shfl_up_sync(0xffffffff, sum, 16);
    if (laneId >= 16) sum += n;
    
    // Il totale del warp si trova ora nell'ultimo thread (lane 31)
    total_warp_sum = __shfl_sync(0xffffffff, sum, 31);
    
    // Ritorniamo l'offset
    return sum - val;
}

__global__ void extract_active_voxels(int* d_voxels, Voxel* d_active_voxels, int* d_num_active_voxels) {

    // Indice base del thread e stride totale (numero totale di thread lanciati)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_vectors = blockDim.x * gridDim.x; // Salto per il prossimo int4

    // Parametri definiti nel params.hpp (ILP=16, READ_WIDTH=4 -> 4 vettori)
    int vectors_per_thread = ILP_FACTOR / READS_PER_THREAD; 

    // --- 1. LETTURA STRIDED (Interleaved) ---
    int voxel_num_points_array[ILP_FACTOR];
    int active_mask[ILP_FACTOR] = {0};
    int local_active_count = 0;

    // Assumiamo che d_voxels sia trattato come array di int4
    int4* base_ptr_int4 = reinterpret_cast<int4*>(d_voxels);

    #pragma unroll
    for (int k = 0; k < vectors_per_thread; k++) {
        // L'indice del vettore salta di 'stride' ad ogni iterazione
        int current_vec_idx = idx + k * stride_vectors;

        if (current_vec_idx < NUM_INT4) {
            // Lettura Coalesced Perfetta
            int4 voxel_quad = base_ptr_int4[current_vec_idx];

            voxel_num_points_array[k*4]     = voxel_quad.x;
            voxel_num_points_array[k*4 + 1] = voxel_quad.y;
            voxel_num_points_array[k*4 + 2] = voxel_quad.z;
            voxel_num_points_array[k*4 + 3] = voxel_quad.w;
        } else {
            // Padding a zero se saltiamo fuori dalla memoria
            voxel_num_points_array[k*4]     = 0;
            voxel_num_points_array[k*4 + 1] = 0;
            voxel_num_points_array[k*4 + 2] = 0;
            voxel_num_points_array[k*4 + 3] = 0;
        }

    }

    // --- 2. FILTRO LOCALE ---
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // Nota: non serve controllare i bounds qui se abbiamo fatto padding a 0 sopra
        // (supponendo MIN_POINTS > 0)
        if (voxel_num_points_array[i] > MIN_POINTS_IN_VOXEL_TO_RENDER) {
            active_mask[i] = 1;
            local_active_count++;
        }
    }

    // --- 3. AGGREGAZIONE WARP ---
    int warp_total_count = 0;
    int my_warp_offset = warpPrefixSum(local_active_count, warp_total_count);
    
    int warp_global_start_idx = 0;
    if ((threadIdx.x % 32) == 0 && warp_total_count > 0) {
        warp_global_start_idx = atomicAdd(d_num_active_voxels, warp_total_count);
    }
    warp_global_start_idx = __shfl_sync(0xffffffff, warp_global_start_idx, 0);
    
    int current_out_idx = warp_global_start_idx + my_warp_offset;

    // --- 4. SCRITTURA ---
    if (local_active_count > 0) {
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            if (active_mask[i]) {
                
                // --- CALCOLO INDICE REALE ---
                // Poiché abbiamo letto a salti (stride), dobbiamo ricostruire l'indice originale.
                // i / 4 -> indica a quale iterazione di caricamento (k) appartiene questo voxel
                // i % 4 -> indica l'offset dentro l'int4
                int k_iteration = i >> 2; // i / 4
                int sub_offset = i & 3;   // i % 4
                
                // Ricostruzione: (IndiceVettore * 4) + sub_offset
                // IndiceVettore = idx + k * stride
                int vec_idx_original = idx + k_iteration * stride_vectors;
                int true_idx = vec_idx_original * 4 + sub_offset;

                // Calcolo coordinate (Standard)
                int temp = true_idx;
                int x = temp % NUM_VOXELS_X;
                temp /= NUM_VOXELS_X;
                int y = temp % NUM_VOXELS_Y;
                int z = temp / NUM_VOXELS_Y;

                short4 voxel_data = make_short4((short)x, (short)y, (short)z, (short)voxel_num_points_array[i]);
                
                reinterpret_cast<short4*>(d_active_voxels)[current_out_idx] = voxel_data;
                current_out_idx++;
            }
        }
    }
}


__global__ void extract_active_voxels_2(int* d_voxels, Voxel* d_active_voxels, int* d_num_active_voxels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_input_idx = idx * ILP_FACTOR;

    // --- 1. LETTURA (Invariata) ---
    // Usiamo variabili locali per evitare accessi spuri se siamo fuori range
    int voxel_num_points_array[ILP_FACTOR];
    int active_mask[ILP_FACTOR] = {0};
    int local_active_count = 0;

    // Check bounds preliminare sicuro
    bool is_valid_thread = (base_input_idx < NUM_TOT_VOXELS);

    if (is_valid_thread) {

        int4 voxel_quad = reinterpret_cast<int4*>(d_voxels)[idx]; 
        voxel_num_points_array[0]=voxel_quad.x;
        voxel_num_points_array[1]=voxel_quad.y;
        voxel_num_points_array[2]=voxel_quad.z;
        voxel_num_points_array[3]=voxel_quad.w;


        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            if (base_input_idx + i < NUM_TOT_VOXELS && voxel_num_points_array[i] > MIN_POINTS_IN_VOXEL_TO_RENDER) {
                active_mask[i] = 1;
                local_active_count++;
            }
        }
    }

    // --- 2. AGGREGAZIONE WARP ADD ---
    
    int warp_total_count = 0;
    // Calcola dove scrivere RELATIVAMENTE all'inizio del blocco del warp
    int my_warp_offset = warpPrefixSum(local_active_count, warp_total_count);
    
    int warp_global_start_idx = 0;
    
    // Solo il primo thread del warp (lane 0) fa la scrittura atomica in memoria
    if ((threadIdx.x % 32) == 0 && warp_total_count > 0) {
        warp_global_start_idx = atomicAdd(d_num_active_voxels, warp_total_count);
    }
    
    // Distribuisce l'indirizzo base globale a tutti i thread del warp
    // __shfl_sync è una lettura effettuata contemporaneamente da tutti i thread attivi nella maschera
    // del valore val preso dal thread con indice 0 all'interno del warp

    warp_global_start_idx = __shfl_sync(0xffffffff, warp_global_start_idx, 0);
    
    // Indice finale dove questo specifico thread inizierà a scrivere
    int current_out_idx = warp_global_start_idx + my_warp_offset;


    // --- 3. SCRITTURA COALESCED ---
    if (is_valid_thread && local_active_count > 0) {
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            if (active_mask[i]) {
                int temp = base_input_idx + i;
                int x = temp % NUM_VOXELS_X;
                temp /= NUM_VOXELS_X;
                int y = temp % NUM_VOXELS_Y;
                int z = temp / NUM_VOXELS_Y;

                short4 voxel_data = make_short4((short)x, (short)y, (short)z, (short)voxel_num_points_array[i]);
                
                // Scrittura all'indirizzo pre-calcolato
                reinterpret_cast<short4*>(d_active_voxels)[current_out_idx] = voxel_data;
                
                // Avanzamento locale (per i prossimi voxel dello stesso thread)
                current_out_idx++;
            }
        }
    }
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

    int* d_voxels_num_points_output;
    Voxel* d_active_voxels;
    int*   d_num_active_voxels;
    // Calcola la dimensione allineata a 4 interi
    int aligned_size = NUM_INT4 * 4;
    CHECK(cudaMalloc(&d_voxels_num_points_output, aligned_size * sizeof(int)));
    CHECK(cudaMalloc(&d_active_voxels, aligned_size * sizeof(Voxel)));
    CHECK(cudaMalloc(&d_num_active_voxels, sizeof(int)));

    Voxel* h_active_voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel));
    int    h_num_active_voxels;

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

        CHECK(cudaMemset(d_voxels_num_points_output, 0, aligned_size * sizeof(int)));

        // LANCIO KERNEL voxelization
        dim3 blockVox(THREAD_BLOCK_SIZE_1D);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        voxelization <<<gridVox, blockVox>>>(d_input, d_voxels_num_points_output, num_points);
        
        // LANCIO KERNEL active_voxels
        CHECK(cudaMemset(d_num_active_voxels, 0, sizeof(int)));
        int num_chunks = (NUM_TOT_VOXELS + ILP_FACTOR - 1) / ILP_FACTOR;
        dim3 blockActiveVoxel(THREAD_BLOCK_SIZE_1D);
        dim3 gridActiveVoxel((num_chunks + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        extract_active_voxels<<<gridActiveVoxel, blockActiveVoxel>>>(d_voxels_num_points_output, d_active_voxels, d_num_active_voxels);


        // COPIA D2H risultati
        CHECK(cudaMemcpy(&h_num_active_voxels, d_num_active_voxels, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_active_voxels, d_active_voxels, h_num_active_voxels * sizeof(Voxel), cudaMemcpyDeviceToHost));


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

    CHECK(cudaFree(d_voxels_num_points_output));
    CHECK(cudaFree(d_active_voxels));
    CHECK(cudaFree(d_num_active_voxels));
    free(h_active_voxels);
    
    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;
}