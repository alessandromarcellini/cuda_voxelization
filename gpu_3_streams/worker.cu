// 3 streams:
//  - 1 H2D
//  - 1 Kernel
//  - 1 D2H

// uso memoria host pinned per trasfermenti + malloc iniziale

// uso di cuda events per sincronizzare gli stream tra loro

// allocazione del numero massimo di punti (stimato se fosse un caso reale) per ospitare i punti ogni tot su device

// più buffer per ospitare i punti sul device, non un solo buffer con offset (è più complesso da gestire e non cambia praticamente niente)
// uso di cuda events per segnalare quando un buffer è stato elaborato e quindi può essere sovrascritto (un evento che segnala che il buffer è libero)
// uso di ringbuffer per gestire l'uso dei buffer

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>

#include "opengl.hpp"
#include "params.hpp"

#define PORT 53456
#define NUM_BUFFERS 3
#define THREAD_BLOCK_SIZE 87
#define MAX_POINTS_PER_BUFFER 128000

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


__global__ void vectorGeneration(float4* d_vectors) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int nx = gridDim.x * blockDim.x;
    int ny = gridDim.y * blockDim.y;
    int idx = z * (nx * ny) + y * nx + x;

    if (x >= NUM_VOXELS_X || y >= NUM_VOXELS_Y || z >= NUM_VOXELS_Z)
        return;

    float4 vector = {
        (x * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_X,
        (y * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Y,
        (z * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Z,
        1.0f
    };

    d_vectors[idx] = vector;

}

__global__ void voxelization(Point* d_input, int* d_output, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point point = d_input[idx];

    // voxelize this point
    int curr_voxel_x = (int)floor((point.x - MIN_X) / DIM_VOXEL);
    int curr_voxel_y = (int)floor((point.y - MIN_Y) / DIM_VOXEL);
    int curr_voxel_z = (int)floor((point.z - MIN_Z) / DIM_VOXEL);
    
    if(curr_voxel_x < 0 || curr_voxel_x >= NUM_VOXELS_X ||
        curr_voxel_y < 0 || curr_voxel_y >= NUM_VOXELS_Y ||
        curr_voxel_z < 0 || curr_voxel_z >= NUM_VOXELS_Z) {
            // punto fuori dai limiti
            return;
    }

    // calcolo indice array lineare voxel
    int voxel_idx = curr_voxel_z * (NUM_VOXELS_X* NUM_VOXELS_Y) + curr_voxel_y * NUM_VOXELS_X + curr_voxel_x;
    
    atomicAdd(&d_output[voxel_idx], 1); 
}



int main(void) {
    //--------------------------------SETUP OPENGL--------------------------------
    // Initialize GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( 1024, 768, WINDOWNAME, NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);


	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // ---------------------------------------------------------------- Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// -----------------------------------------------------------------Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "../opengl_cubes_test/SimpleVertexShader.vertexshader", "../opengl_cubes_test/SimpleFragmentShader.fragmentshader" );


	// The vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
	// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
	static const GLfloat g_vertex_buffer_data[] = {
		-1.0f,-1.0f,-1.0f, // triangle 1 : begin
		-1.0f,-1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f, // triangle 1 : end
		1.0f, 1.0f,-1.0f, // triangle 2 : begin
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f, // triangle 2 : end
		1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		1.0f, 1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		1.0f,-1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f,-1.0f,-1.0f,
		1.0f, 1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f,-1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f,-1.0f, 1.0f
	};

	// Vertices colors
	// One color for each vertex
    float faceGray[6] = {
        0.85f, // face 0 (lighter)
        0.70f,
        0.55f,
        0.40f,
        0.65f,
        0.50f  // face 5 (darker)
    };

    std::vector<GLfloat> g_color_buffer_data;
    g_color_buffer_data.reserve(36 * 3); // 36 vertices RGB
    for (int face = 0; face < 6; face++) {
        float gray = faceGray[face];
        for (int v = 0; v < 6; v++) {
            g_color_buffer_data.push_back(gray); // R
            g_color_buffer_data.push_back(gray); // G
            g_color_buffer_data.push_back(gray); // B
        }
    }

    GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(
        GL_ARRAY_BUFFER,
        g_color_buffer_data.size() * sizeof(float),
        g_color_buffer_data.data(),
        GL_STATIC_DRAW
    );

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);


	// --------------------------------------------------------- Modify camera position
    glm::mat4 Projection = glm::perspective(
        glm::radians(70.0f),
        float(width) / float(height),
        0.1f,
        200.0f
    );

    // Camera at the center of the scene
    glm::vec3 cameraPos(
        (MIN_X + MAX_X / 2) * 0.5f, // 0
        (MIN_Y + MAX_Y) * 0.5f,   // 0
        1.0f                      // camera height
    );

    // Forward direction
    glm::vec3 forward(1.0f, 0.0f, 0.0f);

    // Upward Axis (Z)
    glm::vec3 up(0.0f, 0.0f, 1.0f);

    glm::mat4 View = glm::lookAt(
        cameraPos,
        cameraPos + forward,
        up
    );
	glm::mat4 Model      = glm::mat4(1.0f);
	glm::mat4 MVP        = Projection * View * Model;

    // -------------------------- GENERAZIONE VETTORI TRASLAZIONE --------------------

    float4* vectorTranslations = (float4*) malloc(NUM_TOT_VOXELS * sizeof(float4));
    float4* d_vectors;
    CHECK(cudaMalloc(&d_vectors, NUM_TOT_VOXELS*sizeof(float4)));

    dim3 blockSize(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 gridSize(                           
        (NUM_VOXELS_X + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,           
        (NUM_VOXELS_Y + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,          
        (NUM_VOXELS_Z + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE        
    );

    vectorGeneration <<<gridSize, blockSize>>>(d_vectors);
    CHECK(cudaMemcpy(vectorTranslations, d_vectors, NUM_TOT_VOXELS * sizeof(float4), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_vectors));

    // -------------------------- SETUP SOCKET COMMUNICATION --------------------
    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points;
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


    // ------------------------CUDA STREAMS SETUP -----------------

    cudaStream_t h2d, kernel, d2h;
    CHECK(cudaStreamCreate(&h2d));
    CHECK(cudaStreamCreate(&kernel));
    CHECK(cudaStreamCreate(&d2h));

    Point* h_pinned_inputs[NUM_BUFFERS];
    Point* d_inputs[NUM_BUFFERS];
    int* d_voxels_output[NUM_BUFFERS];
    int* h_voxels_output[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        // alloco memoria host pinned sulla ram
        CHECK(cudaMallocHost((void**)&h_pinned_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));
        // alloco memoria host per output voxels
        CHECK(cudaMallocHost((void**)&h_voxels_output[i], NUM_TOT_VOXELS * sizeof(int)));
        // alloco memoria device
        CHECK(cudaMalloc((void**)&d_inputs[i], MAX_POINTS_PER_BUFFER * sizeof(Point)));
        CHECK(cudaMalloc((void**)&d_voxels_output[i], NUM_TOT_VOXELS * sizeof(int)));
    }

    cudaEvent_t buffer_input_free_events[NUM_BUFFERS];
    cudaEvent_t h2d_done_event[NUM_BUFFERS];
    cudaEvent_t kernel_done_event[NUM_BUFFERS];
    cudaEvent_t buffer_output_done_events[NUM_BUFFERS];
    
    for (int i = 0; i < NUM_BUFFERS; i++) { 
        CHECK(cudaEventCreate(&buffer_input_free_events[i]));
        CHECK(cudaEventCreate(&buffer_output_done_events[i]));
        CHECK(cudaEventCreate(&h2d_done_event[i]));
        CHECK(cudaEventCreate(&kernel_done_event[i]));
        // inizialmente tutti i buffer sono free
        CHECK(cudaEventRecord(buffer_input_free_events[i], 0));
        CHECK(cudaEventRecord(buffer_output_done_events[i], 0));
    }   

    int num_points;
    int i = 0;
    int current_buffer = 0;

    // LOOP RICEZIONE
    while (recv(client_fd, &num_points, sizeof(int), 0) > 0) { 
        
        current_buffer = i % NUM_BUFFERS;
        printf("Ricevuti %d punti da elaborare.\n", num_points);
        
        // 2. FIX RICEZIONE: Ricevi tutto il blocco in una volta
        int total_received = 0;
        int bytes_expected = num_points * sizeof(Point);
        while(total_received < bytes_expected) {
            int received = recv(client_fd, (char*)h_pinned_inputs[current_buffer] + total_received, bytes_expected - total_received, 0);
            if (received <= 0) break; // Errore o chiusura
            total_received += received;
        }

        // --- VOXELIZATION ---

        if (cudaEventQuery(buffer_input_free_events[current_buffer]) != cudaSuccess) {
            // il buffer input non è ancora libero, aspetto
            CHECK(cudaEventSynchronize(buffer_input_free_events[current_buffer]));
        }

        CHECK(cudaMemcpyAsync(d_inputs[current_buffer], h_pinned_inputs[current_buffer], num_points * sizeof(Point), cudaMemcpyHostToDevice, h2d)); 
        CHECK(cudaEventRecord(h2d_done_event[current_buffer], h2d));

        if (cudaEventQuery(buffer_output_done_events[current_buffer]) != cudaSuccess) {
            // il buffer output non è ancora libero, aspetto
            CHECK(cudaEventSynchronize(buffer_output_done_events[current_buffer]));
        }

        CHECK(cudaMemsetAsync(d_voxels_output[current_buffer], 0, NUM_TOT_VOXELS * sizeof(int), d2h)); 

        dim3 blockVox(THREAD_BLOCK_SIZE);
        dim3 gridVox((num_points + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);
        
        if (cudaEventQuery(h2d_done_event[current_buffer]) != cudaSuccess) {
            // l'host to device non è ancora finito, aspetto
            CHECK(cudaEventSynchronize(h2d_done_event[current_buffer]));
        }

        voxelization <<<gridVox, blockVox, 0, kernel>>>(d_inputs[current_buffer], d_voxels_output[current_buffer], num_points);
        CHECK(cudaEventRecord(kernel_done_event[current_buffer], kernel));
        CHECK(cudaEventRecord(buffer_input_free_events[current_buffer], h2d));

        CHECK(cudaMemcpyAsync(h_voxels_output[current_buffer], d_voxels_output[current_buffer], NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost, d2h));
        CHECK(cudaEventRecord(buffer_output_done_events[current_buffer], d2h));

    /*    if (cudaEventQuery(buffer_output_done_events[current_buffer]) == cudaSuccess) {
            
        }   
    */

        printf("Voxelization completata per questo frame.\n");
        i++;

    }
    

}