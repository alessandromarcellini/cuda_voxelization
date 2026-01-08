#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>

#include "opengl.hpp"
#include "params.hpp"

#define RENDERER_PORT 60000


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
    // -------------------------- SETUP SOCKET COMMUNICATION WITH SUPPLIER --------------------
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* d_input;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Error creating socket");
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket");
        exit(1);
    }

    listen(server_fd, 1);
    printf("Server listening on port %d...\n", PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) {
        perror("Error accepting request");
        exit(1);
    }
    printf("Connected with supplier.\n\n");

    // -------------------------- SETUP SOCKET COMMUNICATION WITH RENDERER --------------------
    int renderer_fd;
    struct sockaddr_in renderer_addr;

    renderer_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (renderer_fd < 0) {
        perror("Error creating renderer socket");
        exit(1);
    }

    renderer_addr.sin_family = AF_INET;
    renderer_addr.sin_port = htons(RENDERER_PORT);
    inet_pton(AF_INET, "127.0.0.1", &renderer_addr.sin_addr);

    if (connect(renderer_fd, (struct sockaddr*)&renderer_addr, sizeof(renderer_addr)) < 0) {
        perror("Error connecting to renderer");
        exit(1);
    }

    printf("Connected to renderer on port %d.\n\n", RENDERER_PORT);

    // -------------------------- SETUP VOXELS --------------------------
    Voxel* voxels = (Voxel*) malloc(NUM_TOT_VOXELS * sizeof(Voxel)); // 1D voxel array

    reset_voxels(voxels);

    if (!voxels) {
        perror("Failed to allocate voxel array");
        exit(1);
    }

    // Helper to map 3D indices to 1D
    #define VOXEL_INDEX(i, j, k) ((i) * NUM_VOXELS_Y * NUM_VOXELS_Z + (j) * NUM_VOXELS_Z + (k))

    // -------------------------- SETUP POINTS ------------------------------
    Point* curr_points;
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    int num_points = 0;
    int j = 0;
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) {
        //reset voxels matrix to all zeros
        for (int e = 0; e < NUM_TOT_VOXELS; e++) {
            voxels[e].num_points = 0;
        }
        curr_points = (Point*) malloc(num_points * sizeof(Point));


        //recv points of current frame point cloud
        int total_received = 0;
        int bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected ) {
            int received = recv(client_fd, (char*) curr_points + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

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

        // send voxel array to renderer
        size_t bytes_to_send = NUM_TOT_VOXELS * sizeof(Voxel);
        size_t total_sent = 0;
        while (total_sent < bytes_to_send) {
            printf("ENTERED\n");
            ssize_t sent = send(renderer_fd, ((char*) voxels) + total_sent, bytes_to_send - total_sent, 0);
            if (sent < 0) {
                perror("Error sending voxel data");
                break;
            }
            total_sent += sent;
        }
        printf("MANDATI VOXELS %d\n", j);
        j++;

        free(curr_points);

    }
    free(voxels);
    close(client_fd);
    close(server_fd);
    close(renderer_fd);

	return 0;
}
