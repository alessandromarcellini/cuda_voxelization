#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>

#include "params.hpp"

int compare_names(const void* a, const void* b) {
    const char* name_a = *(const char**)a;
    const char* name_b = *(const char**)b;
    return strcmp(name_a, name_b);
}

int main(void) {
    // INIZIALIZZAZIONE SOCKET
    int sock;
    struct sockaddr_in server_addr;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Errore creazione socket");
        exit(1);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Errore connessione server");
        exit(1);
    }

    // APERTURA CARTELLA, FETCH NOME FILES E SORT
    DIR* dir = opendir(DIRNAME);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", DIRNAME);
        return 1;
    }

    // Fetch all file names and sort them
    struct dirent* entry;
    char* file_names[10000];
    int file_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        file_names[file_count] = strdup(entry->d_name);
        file_count++;
    }
    closedir(dir);

    qsort(file_names, file_count, sizeof(char*), compare_names);

    // ELABORAZIONE FRAME PER FRAME ED INVIO PUNTI
    char path_to_current_frame[512];
    FILE* current_frame;
    float coordinates[FIELDS_PER_POINT];
    Point point;
    int i = 0;

    // --- MODIFICA: Variabile per tracciare il massimo ---
    int max_points_global = 0; 
    char max_points_filename[256];
    // ----------------------------------------------------

    for (int f = 0; f < file_count; f++) {
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

        // --- MODIFICA: Aggiornamento statistica massimo ---
        if (num_points > max_points_global) {
            max_points_global = num_points;
            strcpy(max_points_filename, file_names[f]);
            // Stampa un avviso ogni volta che troviamo un nuovo record (utile per debug immediato)
            printf(">> Nuovo MAX rilevato nel file %s: %d punti\n", file_names[f], max_points_global);
        }
        // --------------------------------------------------

        //invio numero punti
        send(sock, &num_points, sizeof(int), 0);

        // Lettura e invio punti
        while (fread(coordinates, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
            point.x = coordinates[0];
            point.y = coordinates[1];
            point.z = coordinates[2];

            // manda tramite socket
            send(sock, &point, sizeof(Point), 0);
        }
        fclose(current_frame);

        printf("FINITO FILE %s (Punti: %d)\n", file_names[f], num_points);
    }
    
    close(sock);
    
    // Free file name strings
    for (int f = 0; f < file_count; f++) {
        free(file_names[f]);
    }

    // --- MODIFICA: Stampa finale riassuntiva ---
    printf("\n============================================\n");
    printf(" ELABORAZIONE COMPLETATA \n");
    printf("============================================\n");
    printf(" Totale file processati: %d\n", file_count);
    printf(" FILE PIÙ GRANDE: %s\n", max_points_filename);
    printf(" NUMERO MASSIMO PUNTI: %d\n", max_points_global);
    printf("============================================\n");
    printf("Suggerimento per worker.cu:\n");
    printf("#define MAX_POINTS_PER_BUFFER %d\n", max_points_global + 1000); // Un po' di margine
    printf("============================================\n");
    // -------------------------------------------

    return 0;
}