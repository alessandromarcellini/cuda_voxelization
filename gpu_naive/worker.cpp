#include "point.h"


#define BLOCK_SIZE 256

#define CHECK(call){
    const cudaError_t error = call;
        if (error != cudaSuccess){
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error,
        cudaGetErrorString(error));
        exit(1);
    }
}



__global__ void vectorGeneration(float4* d_output,
    int NUM_VOXELS_X, int NUM_VOXELS_Y, int NUM_VOXELS_Z,
    float DIM_VOXEL) {

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    z = blockIdx.z * blockDim.z + threadIdx.z;
    nx = gridDim.x * blockDim.x;
    ny = gridDim.y * blockDim.y;
    int idx = z * (nx * ny) + y * nx + x;

    if (x >= NUM_VOXELS_X || y >= NUM_VOXELS_Y || z >= NUM_VOXELS_Z)
        return;

    float4 vector = {
        x * DIM_VOXEL + DIM_VOXEL / 2.0f,
        y * DIM_VOXEL + DIM_VOXEL / 2.0f,
        z * DIM_VOXEL + DIM_VOXEL / 2.0f,
        1.0f
    };

    d_output[idx] = vector;

}



__global__ void voxelization(float* d_input, int* d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point p = points[idx];


    // voxelize this point
    int curr_voxel_x = (int)floor((point[0] - MIN_X) / DIM_VOXEL);
    int curr_voxel_y = (int)floor((point[1] - MIN_Y) / DIM_VOXEL);
    int curr_voxel_z = (int)floor((point[2] - MIN_Z) / DIM_VOXEL);
    
    if(curr_voxel_x < 0 || curr_voxel_x >= NUM_VOXELS_X ||
        curr_voxel_y < 0 || curr_voxel_y >= NUM_VOXELS_Y ||
        curr_voxel_z < 0 || curr_voxel_z >= NUM_VOXELS_Z) {
            // punto fuori dai limiti
            return;
    }

    // calcolo indice array lineare voxel
    int voxel_idx = (z * grid_dim_y + y) * grid_dim_x + x; curr_voxel_z * (NUM_VOXELS_X* NUM_VOXELS_Y) + curr_voxel_y * NUM_VOXELS_X + curr_voxel_x;
    
    atomicAdd(&d_output[voxel_idx], 1); 
}



int main(void) {
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points, d_input;
    int* d_output;
    int* voxels;

    // -------------------------------- SET TRASLATION VECTORS --------------------------------

    // creo buffer openGL per interop con CUDA

    GLuint voxelOffsetBuffer; // variabile per id buffer openGL
    glGenBuffers(1, &voxelOffsetBuffer); // creo ill buffer e gli assegno un id
    glBindBuffer(GL_ARRAY_BUFFER, voxelOffsetBuffer); // definisco il tipo di buffer

    glBufferData(GL_ARRAY_BUFFER, // alloco memoria per il buffer
                NUM_TOT_VOXELS * sizeof(float4),
                nullptr,            // nessun dato iniziale
                GL_STATIC_DRAW); // buffer raramente modificato

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // registro il buffer su CUDA

    cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterBuffer(
        &cudaResource, // indirizzo variabile che contiene l'handle del buffer
        voxelOffsetBuffer, // id del buffer openGL
        cudaGraphicsRegisterFlagsNone
    );

    // mappo il buffer

    cudaGraphicsMapResources(1, &cudaResource, 0); // lock del buffer su CUDA

    float4* d_output = nullptr;
    size_t size = 0;

    cudaGraphicsResourceGetMappedPointer(
        (void**)&d_output, // il puntatore device (GPU) reale alla memoria del buffer mappato
        &size,
        cudaResource
    );


    // lancio kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(                           
    (NUM_VOXELS_X + BLOCK_SIZE - 1) / BLOCK_SIZE,           
    (NUM_VOXELS_Y + BLOCK_SIZE - 1) / BLOCK_SIZE,          
    (NUM_VOXELS_Z + BLOCK_SIZE - 1) / BLOCK_SIZE        
    );

    vectorGeneration <<<gridSize, blockSize>>>(d_output, NUM_VOXELS_X, NUM_VOXELS_Y, NUM_VOXELS_Z, DIM_VOXEL);
    
    // rilascio risorse CUDA, d'ora in poi lo spazio di memoria diventa un normale buffer openGL
    cudaGraphicsUnmapResources(1, &cudaResource, 0);


    // -----------------------------------------------------------------------------------------------------

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Creazione socket");
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Errore bind");
        exit(1);
    }

    listen(server_fd, 1);
    printf("Server listening on port %d...\n", PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) {
        perror("Errore accept");
        exit(1);
    }
    printf("Client connesso.\n\n");
    int num_points;
    int i = 0;
    for(;;) {
        recv(client_fd, &num_points, sizeof(int), 0);
        // MALLOC
        curr_points = (Point*) malloc(num_points * sizeof(Point));
        printf("RICEZIONE DI %i PUNTI DA SOCKET:\n\n", num_points);

        // RICEZIONE PUNTI
        for (i = 0; i < num_points; i++) {
            recv(client_fd, &curr_points[i], sizeof(Point), 0);
            printf("Received Point: x=%f, y=%f, z=%f\n", curr_points[i].x, curr_points[i].y, curr_points[i].z);
        }


        // ALLOCAZIONE PUNTI
        CHECK(cudaMalloc(&d_input, num_points * sizeof(Point)));
        CHECK(cudaMemcpy(d_input, curr_points, num_points * sizeof(Point), cudaMemcpyHostToDevice)); 
        
        // ALLOCAZIONE VOXELS
        CHECK(cudaMalloc(&d_output, NUM_TOT_VOXELS));
        CHECK(cudaMemset(d_output, 0, NUM_TOT_VOXELS)); 

        // LANCIO KERNEL
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((num_points + blockSize - 1) / BLOCK_SIZE);
        voxelization <<<gridSize, blockSize>>>(d_input, d_output);

        //Copia D2H risultati
        voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
        CHECK(cudaMemcpy(d_output, voxels, NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost));

    }

    close(client_fd);
    close(server_fd);
    return 0;
}