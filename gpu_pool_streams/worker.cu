// N streams

// uso memoria host pinned per trasfermenti + malloc iniziale

// uso di cuda events per sincronizzare gli stream tra loro

// allocazione del numero massimo di punti (stimato se fosse un caso reale) per ospitare i punti ogni tot su device

// più buffer per ospitare i punti sul device, non un solo buffer con offset (è più complesso da gestire e non cambia praticamente niente)
// uso di cuda events per segnalare quando un buffer è stato elaborato e quindi può essere sovrascritto (un evento che segnala che il buffer è libero)
// uso di ringbuffer per gestire l'uso dei buffer


/*

cudaEventRecord(evento, streamA); <-- Piazzi il comando in coda.

Cosa fa il Driver: "Ok, d'ora in poi l'evento si riferisce a questa nuova operazione futura. Stato attuale: PENDING (In Attesa)."

cudaStreamWaitEvent(streamB, evento);

Cosa fa il Driver: "Devo aspettare l'evento. Vedo che è stato appena schedulato (stato Pending).
Blocco lo streamB finché la GPU non esegue il record."


*/


#include <thread>
#include <queue>
#include <iostream>
#include <condition_variable>
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


#ifndef __CUDACC__

extern "C" {
    // Definiamo i prototipi solo per l'IDE per togliere le righe rosse
    __device__ unsigned int __match_any_sync(unsigned int mask, unsigned int value);
    __device__ unsigned int __shfl_sync(unsigned int mask, int var, int srcLane, int width=32);
    __device__ unsigned int __shfl_up_sync(unsigned int mask, int var, unsigned int delta, int width=32);
    
    __device__ int __popc(unsigned int x);
    __device__ int __ffs(int x);
    __device__ int __float2int_rd(float x);
}

#endif

#define THREAD_BLOCK_SIZE_1D 256


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



// __restrict__ dice al compilatore che una certa area di memoria è modificata solo accedendovi con il puntatore ristretto

__global__ void __launch_bounds__(THREAD_BLOCK_SIZE_1D) 
voxelization(Point* __restrict__ d_input, int* __restrict__ d_num_points_output, int num_points) {
    
    // Calcolo indici base
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = idx >> 5;

    // Ogni warp gestisce un blocco contiguo di memoria
    int warp_base = warp_id * TOT_READS_PER_WARP;
    int base_input_idx = warp_base + lane;

    if (warp_base < num_points)
        return;

    const int warp_size = 32;
    const float r_min_x = MIN_X;
    const float r_min_y = MIN_Y;
    const float r_min_z = MIN_Z;
    const float r_inv_dim = INV_DIM_VOXEL;

    const int r_num_vox_x = NUM_VOXELS_X;
    const int r_num_vox_y = NUM_VOXELS_Y;
    const int r_num_vox_z = NUM_VOXELS_Z;

    // REGISTERS PREFETCH: Creiamo un buffer locale nei registri
    Point local_points[ILP_FACTOR];

    // 1. BURST LOAD (Prefetching)
    // Carichiamo TUTTI i dati necessari per questo thread prima di processarli.
    // Questo riempie la pipeline di memoria e riduce gli stalli durante il calcolo.
    // NOTA: Assumiamo che d_input sia "padded" nel main, quindi rimuoviamo il check `current_idx < num_points`
    // Se idx supera num_points reale, leggeremo spazzatura che verrà scartata dal check `inside`.
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // L'istruzione di Load viene emessa qui. La GPU passerà alla prossima istruzione
        // senza aspettare che il dato arrivi, se possibile.
        local_points[i] = d_input[base_input_idx + i * warp_size];
    }

    // 2. COMPUTE & AGGREGATE LOOP
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        
        Point p = local_points[i]; // Dato già nei registri (o in arrivo)

        // Math (ALU) - Ora la pipeline ALU è piena mentre la memoria lavorava prima
        int curr_voxel_x = __float2int_rd((p.x - r_min_x) * r_inv_dim);
        int curr_voxel_y = __float2int_rd((p.y - r_min_y) * r_inv_dim);
        int curr_voxel_z = __float2int_rd((p.z - r_min_z) * r_inv_dim);

        // Controllo limiti (Branch predication friendly)
        bool inside = (curr_voxel_x >= 0 && curr_voxel_x < r_num_vox_x) &&
                      (curr_voxel_y >= 0 && curr_voxel_y < r_num_vox_y) &&
                      (curr_voxel_z >= 0 && curr_voxel_z < r_num_vox_z);

        // Usiamo __any_sync per vedere se ALMENO un thread nel warp deve lavorare.
        // Se tutti i punti del warp sono fuori (es. padding finale), saltiamo tutto il blocco pesante.

            
        // Calcolo indice solo se serve, ma per evitare divergenza conviene calcolarlo dummy
        if(inside) {
            int voxel_idx = curr_voxel_z * (r_num_vox_x * r_num_vox_y) + 
                        curr_voxel_y * r_num_vox_x + 
                        curr_voxel_x;

            // --- WARP AGGREGATION ---
            // Matchiamo solo chi ha un indice valido e uguale
            unsigned int match_mask = __match_any_sync(__activemask(), voxel_idx);

            int aggregation_count = __popc(match_mask);
            int leader_lane = __ffs(match_mask) - 1;

            if (lane == leader_lane) {
                atomicAdd(&d_num_points_output[voxel_idx], aggregation_count);
            }
        }
    }
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


__global__ void extract_active_voxels(int* __restrict__ d_voxels, Voxel* __restrict__ d_active_voxels, int* d_num_active_voxels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int lane = threadIdx.x & 31;
    int warp_id = idx >> 5;

    // base memory index for this warp
    int warp_base = warp_id * TOT_READS_PER_WARP;
    int base_input_idx = warp_base + lane;

    // --- 1. LETTURA (Invariata) ---
    // Usiamo variabili locali per evitare accessi spuri se siamo fuori range
    int voxel_num_points_array[ILP_FACTOR];
    int active_mask[ILP_FACTOR] = {0};
    int local_active_count = 0;

    // Check bounds preliminare sicuro
    bool is_valid_thread = (warp_base < ALIGNED_SIZE_ACTIVE_VOXELS);

    if (is_valid_thread) {
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++){

            voxel_num_points_array[i] = d_voxels[base_input_idx + i*WARP_SIZE];
            if (voxel_num_points_array[i] > MIN_POINTS_IN_VOXEL_TO_RENDER) {
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
                int temp = base_input_idx + i*WARP_SIZE;
                int plane = NUM_VOXELS_X*NUM_VOXELS_Y;
                int z = temp / plane;
                int rem = temp - z*plane;
                int y = rem / NUM_VOXELS_X;
                int x = rem - y*NUM_VOXELS_X;

                short4 voxel_data = make_short4((short)x, (short)y, (short)z, (short)voxel_num_points_array[i]);
                
                // Scrittura all'indirizzo pre-calcolato
                reinterpret_cast<short4*>(d_active_voxels)[current_out_idx] = voxel_data;
                
                // Avanzamento locale (per i prossimi voxel dello stesso thread)
                current_out_idx++;
            }
        }
    }
}


std::queue<CallbackData> full_voxel_buffers;
std::mutex mtx;
std::condition_variable queueCV;
bool stop = false;


void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *data) {

    // Casting del puntatore void* alla nostra struttura
    CallbackData *args = (CallbackData *)data;

    // Controllo errori CUDA precedenti (buona norma)
    if (status != cudaSuccess) {
        printf("Errore stream CUDA prima della callback: %d\n", status);
        // Liberiamo la memoria allocata per gli argomenti prima di uscire
        free(args); 
        return;
    }

    // risveglio del thread sender
    {
        std::lock_guard<std::mutex> lock(mtx);
        full_voxel_buffers.push(*args);
    }
    queueCV.notify_one();

    free(args);
}

bool isEmpty(const CallbackData cb)
{
    return cb.active_count == NULL;;
}

void resetBuffer(CallbackData* buffer)
{
    buffer->active_count = NULL;
}

// ---------------------------------------------------------------------------
void send_voxels(int sock_fd, cudaEvent_t* output_was_sent_events, cudaStream_t signal) {
    // enable bufferization for reordering
    CallbackData buffers[NUM_BUFFERS];
    int next_buffer_to_send = 0;

    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].active_count = NULL;
    }

    do {
        std::unique_lock<std::mutex> lock(mtx);
        queueCV.wait(lock, []{ return !full_voxel_buffers.empty();});

        while(!full_voxel_buffers.empty()) {
            CallbackData cb_data = full_voxel_buffers.front();
            full_voxel_buffers.pop();
            

            // if data is not the next one to send bufferize it
            if (cb_data.buff_id != next_buffer_to_send) {
                buffers[cb_data.buff_id] = cb_data;
                printf("BUFFERIZZATI %d voxel.\n", *cb_data.active_count);
            }
            else {
                lock.unlock(); // sblocco la mutex mentre invio i dati
                // Invio il numero di voxel attivi
                int active_count = *(cb_data.active_count);
                send(sock_fd, &active_count, sizeof(int), 0);

                // Invio i voxel attivi
                int bytes_to_send = active_count * sizeof(Voxel);
                int total_sent = 0;
                while (total_sent < bytes_to_send) {
                    int sent = send(sock_fd, (char*)cb_data.buffer_ptr + total_sent, bytes_to_send - total_sent, 0);
                    if (sent <= 0) {
                        perror("Errore invio dati al renderer");
                        break;
                    }
                    total_sent += sent;
                }
                next_buffer_to_send = (next_buffer_to_send + 1) % NUM_BUFFERS;

                cudaEventRecord(output_was_sent_events[cb_data.buff_id], signal);
                printf("Inviati %d voxel attivi al renderer.\n", active_count);

                // ------------------------------------------------
                // controlla se i prossimi frame sono bufferizzati, in casd send e incremento next_buffer_to_send;
                int i = 0;
                for (int j = next_buffer_to_send; i <= NUM_BUFFERS; j = (j + 1) % NUM_BUFFERS) {

                    if (isEmpty(buffers[j])) {
                        break;
                    }
                    
                    //if buffer content is not null send it and reset it to null
                    // Invio il numero di voxel attivi
                    int active_count = *(buffers[j].active_count);
                    send(sock_fd, &active_count, sizeof(int), 0);

                    // Invio i voxel attivi
                    int bytes_to_send = active_count * sizeof(Voxel);
                    int total_sent = 0;
                    while (total_sent < bytes_to_send) {
                        int sent = send(sock_fd, (char*)buffers[j].buffer_ptr + total_sent, bytes_to_send - total_sent, 0);
                        if (sent <= 0) {
                            perror("Errore invio dati al renderer");
                            break;
                        }
                        total_sent += sent;
                    }
                    next_buffer_to_send = (next_buffer_to_send + 1) % NUM_BUFFERS;
                    resetBuffer(&buffers[j]);
                    cudaEventRecord(output_was_sent_events[buffers[j].buff_id], signal);
                    printf("Inviati %d voxel attivi al renderer BUFFERIZZATI.\n", active_count);
                    i++;
                }
                lock.lock(); // ri-blocco la mutex per controllare la coda
            }
        }

    }while(!stop);

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
    cudaStream_t streams[NUM_BUFFERS];
    cudaStream_t signal;
    CHECK(cudaStreamCreate(&signal));
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }

    // creating NUM_BUFFERS buffers to manage multiple frames a time as inputs
    Point* h_pinned_inputs[NUM_BUFFERS];
    Point* d_inputs[NUM_BUFFERS];

    // creating NUM_BUFFERS buffers for output voxels
    int* d_voxels_output[NUM_BUFFERS];

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
        CHECK(cudaMalloc((void**)&d_inputs[i], ALIGNED_SIZE_VOXELIZATION * sizeof(Point)));

        // alloco memoria device per output
        CHECK(cudaMalloc((void**)&d_voxels_output[i], ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(int)));

        //alloco memoria per contatori e voxel attivi
        CHECK(cudaMalloc((void**)&d_active_voxels[i], ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(Voxel)));
        CHECK(cudaMalloc((void**)&d_num_active_voxels[i], sizeof(int)));

        CHECK(cudaMallocHost((void**)&h_active_voxels[i], MAX_POINTS_PER_BUFFER * sizeof(Voxel)));
    }


    // --------------------EVENTS SETUP --------------------------------
    // events to manage the buffers
    cudaEvent_t h2d_done_events[NUM_BUFFERS];
    cudaEvent_t output_was_sent_events[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++) {
        // creazione eventi
        CHECK(cudaEventCreateWithFlags(&h2d_done_events[i], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&output_was_sent_events[i], cudaEventDisableTiming));

        // Inizializzazione eventi per il primo giro
        CHECK(cudaEventRecord(h2d_done_events[i], streams[i]));
        CHECK(cudaEventRecord(output_was_sent_events[i], signal));
    }

    // ------------------------ SETUP THREAD SENDER-------------------------
    std::thread sender(send_voxels, renderer_fd, output_was_sent_events, signal);

    // -----------------LOOP RICEZIONE----------------------------
    int num_points;
    int i = 0, current_stream = 0, total_received = 0, bytes_expected = 0;

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
    
        CHECK(cudaMemsetAsync(d_voxels_output[current_stream], 0, ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(int), streams[current_stream]));

        int num_chunks = (num_points + ILP_FACTOR - 1) / ILP_FACTOR;
        dim3 blockVox(THREAD_BLOCK_SIZE_1D);
        dim3 gridVox((num_chunks + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
        voxelization <<<gridVox, blockVox, 0, streams[current_stream]>>>(d_inputs[current_stream], d_voxels_output[current_stream], num_points);
        
        CHECK(cudaMemsetAsync(d_num_active_voxels[current_stream], 0, sizeof(int), streams[current_stream]));

        // kernel di compattazione
        num_chunks = (NUM_TOT_VOXELS + ILP_FACTOR - 1) / ILP_FACTOR;
        dim3 blockActiveVoxel(THREAD_BLOCK_SIZE_1D);
        dim3 gridActiveVoxel((num_chunks + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);

        extract_active_voxels<<<gridActiveVoxel, blockActiveVoxel, 0, streams[current_stream]>>>(
            d_voxels_output[current_stream],
            d_active_voxels[current_stream],
            d_num_active_voxels[current_stream]
        );

        CHECK(cudaStreamWaitEvent(streams[current_stream], output_was_sent_events[current_stream], 0));

        //copia a host del numero di voxel attivi
        CHECK(cudaMemcpyAsync(&h_num_active_voxels[current_stream],
                      d_num_active_voxels[current_stream],
                      sizeof(int),
                      cudaMemcpyDeviceToHost,
                      streams[current_stream]));
        //copia a host dei voxel attivi
        CHECK(cudaMemcpyAsync(h_active_voxels[current_stream],
                            d_active_voxels[current_stream],
                            MAX_POINTS_PER_BUFFER * sizeof(Voxel),
                            cudaMemcpyDeviceToHost,
                            streams[current_stream]));

        // Riempimento dati (Socket, Puntatore al buffer specifico, Dimensione, ID)
        
        CallbackData *cb_args = (CallbackData *)malloc(sizeof(CallbackData));
        cb_args->buffer_ptr = h_active_voxels[current_stream];
        cb_args->active_count = &(h_num_active_voxels[current_stream]);
        cb_args->buff_id = current_stream;

        CHECK(cudaStreamAddCallback(streams[current_stream], callback, (void*)cb_args, 0));
        i++;
    }
    
    // CLEANUP

    // shutdown thread sender
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    queueCV.notify_one();
    sender.join();


    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaStreamDestroy(signal));
    }


    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaFreeHost(h_pinned_inputs[i]));
        CHECK(cudaFreeHost(h_active_voxels[i]));
        CHECK(cudaFree(d_inputs[i]));
        CHECK(cudaFree(d_voxels_output[i]));
        CHECK(cudaFree(d_num_active_voxels[i]));
        CHECK(cudaFree(d_active_voxels[i]));
    }

    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaEventDestroy(h2d_done_events[i]));
        CHECK(cudaEventDestroy(output_was_sent_events[i]));
    }

    close(client_fd);
    close(server_fd);
    if(renderer_fd >= 0) close(renderer_fd);

    return 0;

}