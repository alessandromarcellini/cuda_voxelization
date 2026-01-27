
//      SPEEDUP GPU NAIVE


// open 10 files and read them putting them into memory.

//5x esecuzione cpu in cui calcoli il tempo di esecuzione con std::chrono::high_resolution_clock e lo aggiungi ad un accumulatore
// crea la media dei tempi come avg_cpu_time = sum_times_cpu / 5;


// warmup the gpu (3 run prima di quella effettiva in cui si calcoleranno i tempi)
//esegui 5x la roba con la gpu calcolando il tempo di esecuzione con degli eventi start e stop

//Nota: una esecuzione i test elabora tutti i 10 file.
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <chrono>
#include <thread>

#include "../headers/params.hpp"

//supplier stuff
#define NUM_FILES_TO_PROCESS 10

typedef struct {
    int num_points;
    Point* points;
} Frame;

int compare_names(const void* a, const void* b) {
    const char* name_a = *(const char**)a;
    const char* name_b = *(const char**)b;
    return strcmp(name_a, name_b);
}

//CPU worker stuff
#define NUM_TESTS 5

VoxelIndices calculate_voxel_indices(Point point) {
    VoxelIndices result;
    result.i = (int)floor((point.x - MIN_X) / DIM_VOXEL);
    result.j = (int)floor((point.y - MIN_Y) / DIM_VOXEL);
    result.k = (int)floor((point.z - MIN_Z) / DIM_VOXEL);
    return result;
}


void reset_voxels(Voxel* voxels) {
    for (int i = 0; i < NUM_VOXELS_X; i++) {
        for (int j = 0; j < NUM_VOXELS_Y; j++) {
            for (int k = 0; k < NUM_VOXELS_Z; k++) {
                int linear_idx = i * (NUM_VOXELS_Y * NUM_VOXELS_Z) + j * NUM_VOXELS_Z + k;
                voxels[linear_idx].num_points = 0;
                voxels[linear_idx].x = i;
                voxels[linear_idx].y = j;
                voxels[linear_idx].z = k;
            }
        }
    }
}


//GPU STUFF
#define NUM_WARMUPS 3

// Kernel di warmup: tocca ogni elemento moltiplicandolo per 1 (nessun effetto reale)
__global__ void warmupKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 1.0f; // operazione “neutra”
    }
}


// ------------------------------------------------------------------------------------------------------------------------
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


__global__ void extract_active_voxels(int* d_voxels, Voxel* d_active_voxels, int* d_num_active_voxels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int lane = threadIdx.x & 31; // thread idx inside warp
    int warp_id = idx >> 5; // divisione per 32 => id del warp all'interno del blocco

    // base memory index for this warp
    int warp_base = warp_id * (WARP_SIZE * ILP_FACTOR); // indice di memoria del primo elemento che deve gestire il warp
    int base_input_idx = warp_base + lane; // indice di memoria che il thread corrente deve gestire

    // --- 1. LETTURA (Invariata) ---
    // Usiamo variabili locali per evitare accessi spuri se siamo fuori range
    int voxel_num_points_array[ILP_FACTOR];
    int active_mask[ILP_FACTOR] = {0};
    int local_active_count = 0;

    // Check bounds preliminare sicuro
    bool is_valid_thread = (warp_base < NUM_TOT_VOXELS);

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
                //calcolo coordinate del voxel
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




int main(void) {
    printf("================================\n");
    printf("Speedup calculator for GPU NAIVE\n");
    printf("================================\n");
    // open 10 files and read them putting them into memory.
    // APERTURA CARTELLA, FETCH NOME FILES E SORT
    DIR* dir = opendir(DIRNAME);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", DIRNAME);
        return 1;
    }

    // Fetch all file names and sort them
    struct dirent* entry;
    char* all_file_names[10000];
    int file_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        all_file_names[file_count] = strdup(entry->d_name);
        file_count++;
    }
    closedir(dir);

    qsort(all_file_names, file_count, sizeof(char*), compare_names);

    // get only the first 10 file names
    char* file_names[NUM_FILES_TO_PROCESS];
    printf("Getting first 10 files to process:\n");
    for (int i = 0; i < NUM_FILES_TO_PROCESS; i++) {
        file_names[i] = all_file_names[i];
        printf("\t%s\n", file_names[i]);
    }

    // read files and put them in memory
    char path_to_current_frame[512];
    FILE* current_frame;
    float coordinates[FIELDS_PER_POINT];
    Frame frames[NUM_FILES_TO_PROCESS];

    printf("Reading files content and storing it into RAM:\n");
    for (int f = 0; f < NUM_FILES_TO_PROCESS; f++) {
        printf("\tReading file: %s;\n", file_names[f]);
        sprintf(path_to_current_frame, "%s/%s", DIRNAME, file_names[f]);

        // caricamento dati in memoria
        current_frame = fopen(path_to_current_frame, "rb");
        if (current_frame == NULL) {
            perror("Errore apertura file input");
            free(file_names[f]);
            continue;
        }
        //calcolo numero punti
        fseek(current_frame, 0, SEEK_END);
        long file_size = ftell(current_frame);
        fseek(current_frame, 0, SEEK_SET);
        int num_points = file_size / (FIELDS_PER_POINT * sizeof(float));
        frames[f].num_points = num_points;
        frames[f].points = (Point*)malloc(num_points * sizeof(Point));

        // lettura punti dal file
        for (int i = 0; i < num_points; i++) {
            if (fread(coordinates, sizeof(float), FIELDS_PER_POINT, current_frame) != FIELDS_PER_POINT) {
                fprintf(stderr, "Errore lettura punto %d nel file %s\n", i, file_names[f]);
                frames[f].num_points = i;
                return 1;
            }
            frames[f].points[i].x = coordinates[0];
            frames[f].points[i].y = coordinates[1];
            frames[f].points[i].z = coordinates[2];
        }
        fclose(current_frame);
    }

    printf("Starting CPU time testing in 3 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // ------------------------------------------------------- CPU EXECUTION-------------------------------------------------------
    //5x esecuzione cpu in cui calcoli il tempo di esecuzione con std::chrono::high_resolution_clock e lo aggiungi ad un accumulatore
    // crea la media dei tempi come avg_cpu_time = sum_times_cpu / 5;
    Voxel* voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel)); // 1D voxel array
    Voxel* active_voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel)); 
    int active_count = 0;

    double cpu_time_sum_ms = 0.0;
    double gpu_time_sum_ms = 0.0;

    reset_voxels(voxels);

    if (!voxels) {
        perror("Failed to allocate voxel array");
        exit(1);
    }

    // Helper to map 3D indices to 1D
    #define VOXEL_INDEX(i, j, k) ((i) * NUM_VOXELS_Y * NUM_VOXELS_Z + (j) * NUM_VOXELS_Z + (k))

    Point* curr_points;
    Point* curr_points_dummy;

    for(int l = 0; l < NUM_TESTS; l++) {
        printf("\t[CPU] TEST NUMBER %d\n", l +1);
        //calc start time
        auto start_time = std::chrono::high_resolution_clock::now();
        // for each file to process
        for (int k = 0; k < NUM_FILES_TO_PROCESS; k++) {
            int num_points = frames[k].num_points;
            curr_points_dummy = (Point*) malloc(num_points * sizeof(Point));
            curr_points = frames[k].points;

            // voxelize points
            for (int i = 0; i < num_points; i++) {
                VoxelIndices idx = calculate_voxel_indices(curr_points[i]);


                if(idx.i < 0 || idx.i >= NUM_VOXELS_X ||
                    idx.j < 0 || idx.j >= NUM_VOXELS_Y ||
                    idx.k < 0 || idx.k >= NUM_VOXELS_Z) {
                        // point out of bounds
                        continue;
                }

                voxels[VOXEL_INDEX(idx.i, idx.j, idx.k)].num_points++;
            }

            // extract active voxels
            active_count = 0;
            for (int i = 0; i < NUM_TOT_VOXELS; i++) {
                if (voxels[i].num_points > MIN_POINTS_IN_VOXEL_TO_RENDER) {
                    active_voxels[active_count] = voxels[i];
                    active_count++;
                }
            }
            free(curr_points_dummy);
            printf("\t[CPU] Test %d File %d results:\n\tTotal Points Processed: %d,\n\tActive Voxels: %d\n",
                l + 1,
                k + 1,
                frames[k].num_points,
                active_count
            );
        }
        // calc end time
        auto end_time = std::chrono::high_resolution_clock::now();
        // calc time difference
        double curr_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();


        cpu_time_sum_ms += curr_time_ms;

        printf("\t[CPU] Test %d execution time: %.3f ms\n",
            l + 1,
            curr_time_ms
        );
    }

    printf("TOTAL CPU AVG TIME: %.3f\n", cpu_time_sum_ms / NUM_TESTS);



    // ------------------------------------------------------- GPU WARMUP-------------------------------------------------------
    printf("Starting GPU WARMUP in 3 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    // WARM UP
    for (int l = 0; l < NUM_WARMUPS; l++) {
        const int N = 1 << 20; // 1 milione di elementi per warmup
        size_t bytes = N * sizeof(float);

        // Allocazione GPU
        float* d_data;
        cudaMalloc(&d_data, bytes);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // Lancia il kernel di warmup
        warmupKernel<<<gridSize, blockSize>>>(d_data, N);

        // Assicura che il kernel finisca
        cudaDeviceSynchronize();

        
        cudaFree(d_data);
    }
    
    printf("GPU warmup completato!\n");
    printf("Starting GPU time testing in 3 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // ------------------------------------------------------- GPU EXECUTION-------------------------------------------------------

    
    //5x esecuzione cpu in cui calcoli il tempo di esecuzione con std::chrono::high_resolution_clock e lo aggiungi ad un accumulatore
    // crea la media dei tempi come avg_cpu_time = sum_times_cpu / 5;

    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // inserisci un evento start nello stream, fai eseguire tutto normalmente, inserisci un evento stop nello stream, sincronizza a stop e calcola elapsed time
    // Timer CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //setup
    Point* d_input;
    int num_points = 0;

    int* d_voxels_num_points_output;
    Voxel* d_active_voxels;
    int*   d_num_active_voxels;
    int*    h_num_active_voxels;
    Voxel* h_active_voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel));

    // cudaMalloc() garantisce allineamento almeno ad un indirizzo multiplo di 256B
    CHECK(cudaMallocHost((void**)&curr_points, MAX_POINTS_PER_BUFFER * sizeof(Point)));
    CHECK(cudaMalloc(&d_input, ALIGNED_SIZE_VOXELIZATION * sizeof(Point)));
    //CHECK(cudaMemset(d_input, 0, ALIGNED_SIZE_VOXELIZATION * sizeof(Point)));
    CHECK(cudaMalloc(&d_voxels_num_points_output, ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(int)));
    CHECK(cudaMalloc(&d_active_voxels, ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(Voxel)));

    // memoria zero-copy per il numero di voxel attivi
    // Alloca memoria host pinned e mappata
    h_num_active_voxels = (int*)malloc(sizeof(int));
    // Ottieni il puntatore device alla stessa memoria
    cudaMalloc(&d_num_active_voxels, sizeof(int));

    for(int l = 0; l < NUM_TESTS; l++) {
        printf("\t[GPU] TEST NUMBER %d\n", l +1);
        //calc start time
        cudaEventRecord(start);
        // for each file to process
        for (int k = 0; k < NUM_FILES_TO_PROCESS; k++) {
            // put in here computations that are inside the while loop on worker.cu of gpu naive

            // curr_points = frames[k].points;
            memcpy(curr_points, frames[k].points, num_points * sizeof(Point));

            CHECK(cudaMemcpy(d_input,
                curr_points,
                num_points * sizeof(Point),
                cudaMemcpyHostToDevice
            ));
            num_points = frames[k].num_points;
            CHECK(cudaMemcpy(d_input, curr_points, num_points * sizeof(Point), cudaMemcpyHostToDevice)); 

            CHECK(cudaMemset(d_voxels_num_points_output, 0, ALIGNED_SIZE_ACTIVE_VOXELS * sizeof(int)));
            
            // LANCIO KERNEL voxelization
            int num_chunks = (num_points + ILP_FACTOR - 1) / ILP_FACTOR;
            dim3 blockVox(THREAD_BLOCK_SIZE_1D);
            dim3 gridVox((num_chunks + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
            voxelization<<<gridVox, blockVox>>>(d_input, d_voxels_num_points_output, num_points);
            
            // LANCIO KERNEL active_voxels
            cudaMemset(d_num_active_voxels, 0, sizeof(int));
            num_chunks = (NUM_TOT_VOXELS + ILP_FACTOR - 1) / ILP_FACTOR;
            dim3 blockActiveVoxel(THREAD_BLOCK_SIZE_1D);
            dim3 gridActiveVoxel((num_chunks + THREAD_BLOCK_SIZE_1D - 1) / THREAD_BLOCK_SIZE_1D);
            extract_active_voxels<<<gridActiveVoxel, blockActiveVoxel>>>(d_voxels_num_points_output, d_active_voxels, d_num_active_voxels);
            cudaMemcpy(h_num_active_voxels, d_num_active_voxels, sizeof(int), cudaMemcpyDeviceToHost);

            // COPIA D2H risultati
            CHECK(cudaMemcpy(h_active_voxels, d_active_voxels, (*h_num_active_voxels) * sizeof(Voxel), cudaMemcpyDeviceToHost));            
        }        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_total_time;
        cudaEventElapsedTime(&gpu_total_time, start, stop);
        gpu_time_sum_ms += gpu_total_time;
    }


    printf("TOTAL GPU AVG TIME: %.3f\n", gpu_time_sum_ms / NUM_TESTS);

    printf("================================\n");
    printf("=========TOTAL SPEEDUP==========\n");
    printf("================================\n");

    printf("%.3fx\n", (cpu_time_sum_ms / NUM_TESTS) / (gpu_time_sum_ms / NUM_TESTS));


    CHECK(cudaFreeHost(curr_points));
    free(h_num_active_voxels);
    cudaFree(d_num_active_voxels);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_voxels_num_points_output));
    CHECK(cudaFree(d_active_voxels));
    free(h_active_voxels);

    // cleanup
    for (int f = 0; f < NUM_FILES_TO_PROCESS; f++) {
        free(frames[f].points);
    }

}
