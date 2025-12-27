#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FIELDS_PER_POINT 4

#define MAX_X 34
#define MIN_X -34

#define MAX_Y 22
#define MIN_Y -22

#define MAX_Z 3
#define MIN_Z -3

#define DIM_VOXEL 0.1

#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))



typedef struct {
  float x;
  float y;
  float z;
} Point;


int main(void) {
    // Point* points;
    int ***voxels = malloc(NUM_VOXELS_X * sizeof(int**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxels[i] = malloc(NUM_VOXELS_Y * sizeof(int*));
        for(int j=0; j<NUM_VOXELS_Y; j++)
            voxels[i][j] = calloc(NUM_VOXELS_Z, sizeof(int));
    }
    
    char* dir_name = "new_dataset";

    DIR* dir = opendir(dir_name);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", dir_name);
        return 1;
    }
    
    //per ogni frame
    char path_to_current_frame[512];
    struct dirent* entry;
    FILE* current_frame;
    float point[FIELDS_PER_POINT];
    int i = 0;

    int curr_voxel_x;
    int curr_voxel_y;
    int curr_voxel_z;

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        sprintf(path_to_current_frame, "%s/%s", dir_name, entry->d_name);

        // caricamento dati in memoria
        current_frame = fopen(path_to_current_frame, "rb");
        if (current_frame == NULL) {
            perror("Errore apertura file input");
            continue;
        }
        //calcolo numero punti
        fseek(current_frame, 0, SEEK_END);
        long file_size = ftell(current_frame);
        fseek(current_frame, 0, SEEK_SET);
        int num_points = file_size / (FIELDS_PER_POINT * sizeof(float));

        //malloc per allocate array di punti
        // points = (Point*)malloc(num_points * sizeof(Point));
        // if (points == NULL) {
        //     perror("Errore allocazione memoria");
        //     fclose(current_frame);
        //     continue;
        // }

        // Lettura
        while (fread(point, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
                // points[i].x = point[0];
                // points[i].y = point[1];
                // points[i].z = point[2];

                // trova il voxel in cui è
                curr_voxel_x = (int)floor((point[0] - MIN_X) / DIM_VOXEL);
                curr_voxel_y = (int)floor((point[1] - MIN_Y) / DIM_VOXEL);
                curr_voxel_z = (int)floor((point[2] - MIN_Z) / DIM_VOXEL);

                if(curr_voxel_x < 0 || curr_voxel_x >= NUM_VOXELS_X ||
                    curr_voxel_y < 0 || curr_voxel_y >= NUM_VOXELS_Y ||
                    curr_voxel_z < 0 || curr_voxel_z >= NUM_VOXELS_Z) {
                        // punto fuori dai limiti
                        continue;
                }

                voxels[curr_voxel_x][curr_voxel_y][curr_voxel_z]++;
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