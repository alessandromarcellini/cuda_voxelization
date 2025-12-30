#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>


#include "params.hpp"
#define PORT 53456

#define THREAD_BLOCK_SIZE 256

#define CHECK(call){
    const cudaError_t error = call;
        if (error != cudaSuccess){
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error,
        cudaGetErrorString(error));
        exit(1);
    }
}

__global__ void vectorGeneration(float4* d_output) {

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


    int server_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    Point* curr_points, d_input;

    // -------------------------------- SET TRASLATION VECTORS --------------------------------

    // Creating openGL buffer and register it with CUDA

    // GLuint voxelOffsetBuffer; // variabile per id buffer openGL
    // glGenBuffers(1, &voxelOffsetBuffer); // creo il buffer e gli assegno un id
    // glBindBuffer(GL_ARRAY_BUFFER, voxelOffsetBuffer); // definisco il tipo di buffer

    // glBufferData(GL_ARRAY_BUFFER, // alloco memoria per il buffer
    //             NUM_TOT_VOXELS * sizeof(float4),
    //             nullptr,            // nessun dato iniziale
    //             GL_STATIC_DRAW); // buffer raramente modificato

    // glBindBuffer(GL_ARRAY_BUFFER, 0);

    // // registro il buffer su CUDA

    // cudaGraphicsResource* cudaResource;
    // cudaGraphicsGLRegisterBuffer(
    //     &cudaResource, // indirizzo variabile che contiene l'handle del buffer
    //     voxelOffsetBuffer, // id del buffer openGL
    //     cudaGraphicsRegisterFlagsNone
    // );

    // // mappo il buffer
    // cudaGraphicsMapResources(1, &cudaResource, 0); // lock del buffer su CUDA

    // float4* d_output = nullptr;
    // size_t size = 0;

    // cudaGraphicsResourceGetMappedPointer(
    //     (void**)&d_output, // il puntatore device (GPU) reale alla memoria del buffer mappato
    //     &size,
    //     cudaResource
    // );


    // lancio kernel

    float4 vectorTranslations[NUM_TOT_VOXELS];
    float4* d_output;
    CHECK(cudaMalloc(&d_output, NUM_TOT_VOXELS*sizeof(float4)));

    dim3 blockSize(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 gridSize(                           
        (NUM_VOXELS_X + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,           
        (NUM_VOXELS_Y + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,          
        (NUM_VOXELS_Z + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE        
    );

    vectorGeneration <<<gridSize, blockSize>>>(d_output);
    CHECK(cudaMemcpy(d_output, vectorTranslations, NUM_TOT_VOXELS * sizeof(float4), cudaMemcpyDeviceToHost));
    cudaFree(d_output);
    // rilascio risorse CUDA, d'ora in poi lo spazio di memoria diventa un normale buffer openGL
    //cudaGraphicsUnmapResources(1, &cudaResource, 0);


    // --------------------------SETUP SOCKET COMMUNICATION--------------------

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
    int num_points;
    int i = 0;

    // ------------------------FRAME BY FRAME COMPUTATIONS-----------------
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    Point point;
    float lastFrameTime = glfwGetTime();
    bool time_to_advance_frame;
    bool received_num_points = false;
    int num_points_recvd = 0;
    int num_points = 0;

    // coda di numero di punti
    // coda di punti con più punti di frame diversi

    // recv di numero punti -> recv Point e count_recvd++ -> recv Point e count_recvd++
    // se time_to_advance => recv Point num_points - count_recvd => computation


    do {
        //--------------------CHECK IF IT'S TIME TO UPDATE---------------------------
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;

        time_to_advance_frame = false;
        if (deltaTime >= FRAMEDURATION) {
            time_to_advance_frame = true;
            lastFrameTime = currentTime;
        }

        // -----------------"ASYNC" recv while not computing-------------------------
        if (!received_num_points) {
            recv(client_fd, &num_points, sizeof(int), 0);
            curr_points = (Point*) malloc(num_points * sizeof(Point));
            received_num_points = true;
        }
        if (received_num_points && num_points_recvd < num_points) {
            recv(client_fd, &curr_points[num_points_recvd], sizeof(Point), 0);
            num_points_recvd++;
        }

        
        // ----------------------------UPDATE VOXEL DATA----------------------------
        if (time_to_advance_frame) {
            // reset voxels data to all zeros
            memset(voxels, 0, NUM_TOT_VOXELS * sizeof(int));
            // recv ultimi punti rimasti
            for (; num_points_recvd < num_points; num_points_recvd++) {
                recv(client_fd, &curr_points[num_points_recvd], sizeof(Point), 0);
            }

            received_num_points = false;
            num_points_recvd = 0;

            //

            // -----------------------VOXELIZATION-------------------------------
            // ALLOCAZIONE PUNTI
            CHECK(cudaMalloc(&d_input, num_points * sizeof(Point)));
            CHECK(cudaMemcpy(d_input, curr_points, num_points * sizeof(Point), cudaMemcpyHostToDevice)); 
            
            // ALLOCAZIONE VOXELS
            CHECK(cudaMalloc(&d_output, NUM_TOT_VOXELS));
            CHECK(cudaMemset(d_output, 0, NUM_TOT_VOXELS)); 

            // LANCIO KERNEL
            dim3 blockSize(THREAD_BLOCK_SIZE);
            dim3 gridSize((num_points + blockSize - 1) / THREAD_BLOCK_SIZE);
            voxelization <<<gridSize, blockSize>>>(d_input, d_output);
            //Copia D2H risultati
            voxels = (int*) malloc(NUM_TOT_VOXELS * sizeof(int));
            CHECK(cudaMemcpy(d_output, voxels, NUM_TOT_VOXELS * sizeof(int), cudaMemcpyDeviceToHost));
            
            //cleanUP
            cudaFree(d_input);
            cudaFree(d_output);
            free(curr_points);
            

        //---------------------------- RENDER ----------------------------
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);

        GLuint MatrixID = glGetUniformLocation(programID, "MVP");

        // 1st attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0
        );

        // 2nd attribute buffer : colors
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0
        );

        //---------------------render current voxels-------------------------------
        for (int i = 0; i < NUM_TOT_VOXELS; i++) {
            if (voxels[i] > MIN_POINTS_IN_VOXEL_TO_RENDER) {
                //render it
                glm::mat4 Model1 = glm::translate(glm::mat4(1.0f), vectorTranslations[i]);
                Model1 = glm::scale(Model1, glm::vec3(DIM_VOXEL / 2.0f));
                glm::mat4 MVP1 = Projection * View * Model1;
                glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP1[0][0]);
                glDrawArrays(GL_TRIANGLES, 0, 12*3);
            }
        }

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

    free(voxels);
    free(voxelTranslationVectors);
    
    close(client_fd);
    close(server_fd);

    return 0;
}