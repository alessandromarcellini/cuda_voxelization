#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define FIELDS_PER_POINT 4

// Limiti percentili
#define MAX_X 34
#define MAX_Y 22
#define MAX_Z 3
#define MIN_X -34
#define MIN_Y -22
#define MIN_Z -3

int main(void) {
    const char *input_folder = "./dataset";
    const char *output_folder = "./new_dataset";

    DIR *dir;
    struct dirent *entry;

    dir = opendir(input_folder);
    if (dir == NULL) {
        perror("Errore apertura cartella input");
        return EXIT_FAILURE;
    }

    while ((entry = readdir(dir)) != NULL) {
        // Salta "." e ".."
        if (entry->d_type != DT_REG) continue; // solo file regolari
        if (strstr(entry->d_name, ".bin") == NULL) continue; // solo .bin

        // Costruisci path completo input/output
        char path_in[512];
        char path_out[512];
        snprintf(path_in, sizeof(path_in), "%s/%s", input_folder, entry->d_name);
        snprintf(path_out, sizeof(path_out), "%s/%s", output_folder, entry->d_name);

        FILE *file_in = fopen(path_in, "rb");
        if (file_in == NULL) {
            perror("Errore apertura file input");
            continue;
        }

        FILE *file_out = fopen(path_out, "wb");
        if (file_out == NULL) {
            perror("Errore apertura file output");
            fclose(file_in);
            continue;
        }

        float point[FIELDS_PER_POINT];
        int count = 0;

        // Lettura e filtro dei punti
        while (fread(point, sizeof(float), FIELDS_PER_POINT, file_in) == FIELDS_PER_POINT) {
            if (point[0] >= MIN_X && point[0] <= MAX_X &&
                point[1] >= MIN_Y && point[1] <= MAX_Y &&
                point[2] >= MIN_Z && point[2] <= MAX_Z) {

                fwrite(point, sizeof(float), FIELDS_PER_POINT, file_out);
                count++;
            }
        }

        fclose(file_in);
        fclose(file_out);

        printf("File '%s' filtrato, punti rimasti: %d\n", entry->d_name, count);
    }

    closedir(dir);
    return EXIT_SUCCESS;
}
