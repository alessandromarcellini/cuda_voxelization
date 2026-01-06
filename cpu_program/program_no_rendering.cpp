#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <glm/glm.hpp>
#include <time.h>
#include "params.hpp"

int compare_names(const void* a, const void* b) {
    const char* name_a = *(const char**)a;
    const char* name_b = *(const char**)b;
    return strcmp(name_a, name_b);
}

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
    // --------------------------SETUP TRANSLATION VECTORS--------------------
    // initializing rendering voxel space

    glm::vec3 ***voxelTranslationVectors = (glm::vec3***)malloc(NUM_VOXELS_X * sizeof(glm::vec3**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxelTranslationVectors[i] = (glm::vec3**) malloc(NUM_VOXELS_Y * sizeof(glm::vec3*));
        for(int j=0; j < NUM_VOXELS_Y; j++)
            voxelTranslationVectors[i][j] = (glm::vec3*) calloc(NUM_VOXELS_Z, sizeof(glm::vec3)); // allocates memory and writes all bytes to 0
    }
    
    double start_time = clock();

    for (int x = 0; x < NUM_VOXELS_X; ++x) {
        for (int y = 0; y < NUM_VOXELS_Y; ++y) {
            for (int z = 0; z < NUM_VOXELS_Z; ++z) {
                glm::vec3 translation(
                    (x * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_X,
                    (y * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Y,
                    (z * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Z
                );

                voxelTranslationVectors[x][y][z] = translation;
            }
        }
    }

    printf("Finished setting up voxel translation vectors : %lf clocks\n", clock() - start_time);

    // --------------------------SETUP VOXELS MATRIX--------------------------
    int ***voxels = (int***) malloc(NUM_VOXELS_X * sizeof(int**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxels[i] = (int**) malloc(NUM_VOXELS_Y * sizeof(int*));
        for(int j=0; j<NUM_VOXELS_Y; j++)
            voxels[i][j] = (int*) calloc(NUM_VOXELS_Z, sizeof(int)); // allocates memory and writes all bytes to 0
    }

    // APERTURA CARTELLA, FETCH NOME FILES E SORT
    DIR* dir = opendir(DIRNAME);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", DIRNAME);
        return 1;
    }

    // Fetch all file names and sort them
    struct dirent* entry;
    char* file_names[10000]; // TODO make this better
    int file_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        file_names[file_count] = strdup(entry->d_name);
        file_count++;
        if (file_count == 10000) break;
    }
    closedir(dir);

    qsort(file_names, file_count, sizeof(char*), compare_names);

    // FRAME BY FRAME COMPUTATIONS
    char path_to_current_frame[512];
    FILE* current_frame;
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    char* fname;
    float point[FIELDS_PER_POINT];
    int num_frame = 0;

    while(num_frame < file_count) {
        // reset voxels data to all zeros
        for(int i = 0; i < NUM_VOXELS_X; i++) {
            for(int j = 0; j < NUM_VOXELS_Y; j++) {
                memset(voxels[i][j], 0, NUM_VOXELS_Z * sizeof(int));
            }
        }

        fname = file_names[num_frame];
            sprintf(path_to_current_frame, "%s/%s", DIRNAME, fname);
            // opening frame file
            current_frame = fopen(path_to_current_frame, "rb");
            if (current_frame == NULL) {
                perror("ERROR opening frame file in read mode.");
                continue;
            }

        // load frame data
        while (fread(point, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
            // for each point find in which voxel it is
            VoxelIndices curr_voxel_indices = calculate_voxel_indices(point);


            if(curr_voxel_indices.i < 0 || curr_voxel_indices.i >= NUM_VOXELS_X ||
                curr_voxel_indices.j < 0 || curr_voxel_indices.j >= NUM_VOXELS_Y ||
                curr_voxel_indices.k < 0 || curr_voxel_indices.k >= NUM_VOXELS_Z) {
                    // point out of bounds
                    continue;
            }

            voxels[curr_voxel_indices.i][curr_voxel_indices.j][curr_voxel_indices.k]++;
        }
        fclose(current_frame);

        printf("FINISHED COMPUTING FILE: %s\n", fname);
        num_frame++;
    }

    for (int i = 0; i < NUM_VOXELS_X; i++) {
        for (int j = 0; j < NUM_VOXELS_Y; j++) {
            free(voxels[i][j]);
        }
        free(voxels[i]);
    }
    free(voxels);
    

    for (int i = 0; i < NUM_VOXELS_X; i++) {
        for (int j = 0; j < NUM_VOXELS_Y; j++) {
            free(voxelTranslationVectors[i][j]);
        }
        free(voxelTranslationVectors[i]);
    }
    free(voxelTranslationVectors);

	return 0;
}
