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



    // ------------------------------------------------------- CPU EXECUTION-------------------------------------------------------
    //5x esecuzione cpu in cui calcoli il tempo di esecuzione con std::chrono::high_resolution_clock e lo aggiungi ad un accumulatore
    // crea la media dei tempi come avg_cpu_time = sum_times_cpu / 5;

    printf("Starting GPU time testing in 3 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // std::this_thread::sleep_for(std::chrono::seconds(10));

    for (int f = 0; f < NUM_FILES_TO_PROCESS; f++) {
        free(frames[f].points);
    }

}
