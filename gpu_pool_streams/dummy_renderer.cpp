#include <chrono>
#include <cstdio>
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include "../headers/params.hpp"

#define THREAD_BLOCK_SIZE 8

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




float getTimeSeconds() {
    using clock = std::chrono::high_resolution_clock;
    static auto startTime = clock::now();

    auto now = clock::now();
    std::chrono::duration<float> elapsed = now - startTime;
    return elapsed.count(); // secondi
}


int main() {

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
    addr.sin_port = htons(RENDERER_PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket");
        exit(1);
    }

    listen(server_fd, 1);
    printf("Renderer listening on port %d...\n", RENDERER_PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) {
        perror("Error accepting request");
        exit(1);
    }
    printf("Connected with Worker.\n\n");

    // -------------------------- RENDER LOOP --------------------
    float lastFrameTime = getTimeSeconds();
    bool time_to_advance_frame;
    int num_points = 0;
    bool socket_closed = false;
    Voxel* active_voxels = NULL;
    int active_count = 0;

    do {
        //--------------------CHECK IF IT'S TIME TO UPDATE---------------------------
      float currentTime = getTimeSeconds();
      float deltaTime = currentTime - lastFrameTime;

      printf("Frame Rate: %.4f /seconds. \n", 1.0/deltaTime);

      lastFrameTime = currentTime; 

      active_count = 0;
      int received_count = recv(client_fd, &active_count, sizeof(int), 0);
      
      if (received_count <= 0) {
          printf("Connessione chiusa o errore.\n");
          socket_closed = true;
          break;
      }

      active_voxels = (Voxel*) malloc(active_count * sizeof(Voxel));

      int bytes_expected = active_count * sizeof(Voxel);
      int total_received = 0;
      char* ptr_buffer = (char*)active_voxels; // Importante: cast a char* per aritmetica dei puntatori   
      
      while (total_received < bytes_expected) {
          int received = recv(client_fd, ptr_buffer + total_received, bytes_expected - total_received, 0);
          
          if (received == 0) {
              printf("Connessione chiusa dal worker.\n");
              socket_closed = true;
              break;
          }
          if (received < 0) {
              perror("Errore recv");
              break;
          }
          total_received += received;
      }

    } // Check if the ESC key was pressed or the window was closed
	  while(!socket_closed);

    free(active_voxels);
    close(client_fd);


}