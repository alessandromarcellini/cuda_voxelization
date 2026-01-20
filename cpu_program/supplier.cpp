#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include "../headers/params.hpp"

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
    server_addr.sin_port = htons(WORKER_PORT);
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
    //per ogni frame
    char path_to_current_frame[512];
    FILE* current_frame;
    Point* point_buffer = (Point*) malloc(POINT_BUFFER_DIM * sizeof(Point));
    int i = 0, file_size = 0, total_bytes_sent = 0, el_read = 0, chunk_sent = 0, bytes_to_send = 0;;

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
        int num_points = file_size / sizeof(Point);

        //invio numero punti
        printf("Invio %d punti da elaborare al worker\n", num_points);
        send(sock, &num_points, sizeof(int), 0);

        total_bytes_sent = 0, bytes_to_send = 0;
        

        while (total_bytes_sent < file_size) {
            // Leggi dal file
            el_read = fread(point_buffer, sizeof(Point), POINT_BUFFER_DIM, current_frame);
            bytes_to_send = el_read * sizeof(Point);

            if (bytes_to_send <= 0) { // evito di inviare byte spuri
                break;
            }

            // Invio effettivo
            chunk_sent = 0;
            while (chunk_sent < bytes_to_send) {
                int s = send(sock, ((char*)point_buffer) + chunk_sent, bytes_to_send - chunk_sent, 0);
                if (s < 0) { perror("Errore send"); exit(1); }
                chunk_sent += s;
            }
            total_bytes_sent += chunk_sent;
        }

        fclose(current_frame);
        printf("FINITO FILE %s\n", file_names[f]);
    }

    free(point_buffer);
    close(sock);
    
    // Free file name strings
    for (int f = 0; f < file_count; f++) {
        free(file_names[f]);
    }

    return 0;
}