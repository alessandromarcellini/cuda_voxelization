
#include "params.h"

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
    DIR* dir = opendir(DIR_NAME);
    if (dir == NULL) {
        printf("Errore: cartella '%s' non trovata\n", DIR_NAME);
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
    float coordinates[FIELDS_PER_POINT];
    Point point;
    int i = 0;

    for (int f = 0; f < file_count; f++) {
        sprintf(path_to_current_frame, "%s/%s", DIR_NAME, file_names[f]);

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

        //invio numero punti
        send(sock, &num_points, sizeof(int), 0);

        // Lettura
        while (fread(coordinates, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
            point.x = coordinates[0];
            point.y = coordinates[1];
            point.z = coordinates[2];

            // manda tramite socket
            send(sock, &point, sizeof(Point), 0);
            }
            fclose(current_frame);

            printf("FINITO FILE %s\n", file_names[f]);
    }
    close(sock);
    
    // Free file name strings
    for (int f = 0; f < file_count; f++) {
        free(file_names[f]);
    }

    return 0;
}