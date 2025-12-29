#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FIELDS_PER_POINT 4

#define MAX_X 300 //34
#define MAX_Y 200 // 22
#define MAX_Z 100 // 3
#define MIN_X -300 //-34
#define MIN_Y -200 // -22
#define MIN_Z -100 // -3

#define DIM_VOXEL 0.1

#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))

#define DIRNAME "../new_dataset"



typedef struct {
  float x;
  float y;
  float z;
} Point;

typedef struct {
  int i;
  int j;
  int k;
} VoxelIndices;


int calculate_num_points(FILE* file) {
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    return file_size / (FIELDS_PER_POINT * sizeof(float));
}

VoxelIndices calculate_voxel_indices(float* point) {
    VoxelIndices result;
    result.i = (int)floor((point[0] - MIN_X) / DIM_VOXEL);
    result.j = (int)floor((point[1] - MIN_Y) / DIM_VOXEL);
    result.k = (int)floor((point[2] - MIN_Z) / DIM_VOXEL);
    return result;
}


int main(void) {
    // SETUP VOXELS MATRIX
    int ***voxels = (int ***) malloc(NUM_VOXELS_X * sizeof(int**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxels[i] = (int **) malloc(NUM_VOXELS_Y * sizeof(int*));
        for(int j=0; j<NUM_VOXELS_Y; j++)
            voxels[i][j] = (int *) calloc(NUM_VOXELS_Z, sizeof(int)); // allocates memory and writes all bytes to 0
    }

    DIR* dir = opendir(DIRNAME);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", DIRNAME);
        return 1;
    }
    
    // PER OGNI FRAME FAI LE COMPUTAZIONI NECESSARIE
    char path_to_current_frame[512];
    struct dirent* entry;
    FILE* current_frame;
    float point[FIELDS_PER_POINT];
    int i = 0;

    int curr_voxel_x;
    int curr_voxel_y;
    int curr_voxel_z;

    while ((entry = readdir(dir)) != NULL) {
        // skip cartelle . e ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        sprintf(path_to_current_frame, "%s/%s", DIRNAME, entry->d_name);

        // caricamento dati in memoria
        current_frame = fopen(path_to_current_frame, "rb");
        if (current_frame == NULL) {
            perror("Errore apertura file input");
            continue;
        }
        int num_points = calculate_num_points(current_frame);

        // Lettura
        while (fread(point, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
            // trova il voxel in cui è
            VoxelIndices curr_voxel_indices = calculate_voxel_indices(point);


            if(curr_voxel_indices.i < 0 || curr_voxel_indices.i >= NUM_VOXELS_X ||
                curr_voxel_indices.j < 0 || curr_voxel_indices.j >= NUM_VOXELS_Y ||
                curr_voxel_indices.k < 0 || curr_voxel_indices.k >= NUM_VOXELS_Z) {
                    // punto fuori dai limiti
                    continue;
            }

            voxels[curr_voxel_indices.i][curr_voxel_indices.j][curr_voxel_indices.k]++;
            }
            fclose(current_frame);

            printf("FINITO FILE %s\n", entry->d_name);

            // render grafico


            // Reinizializza voxels a tutti zeri
            for(int i = 0; i < NUM_VOXELS_X; i++) {
                for(int j = 0; j < NUM_VOXELS_Y; j++) {
                    memset(voxels[i][j], 0, NUM_VOXELS_Z * sizeof(int));
                }
            }   


        // free array di punti
        // free(points);
    }
    
    closedir(dir);

    return 0;

}
