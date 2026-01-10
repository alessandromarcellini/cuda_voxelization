#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

#include "opengl.hpp"
#include "params.hpp"

#define MAX_DENSITY_THRESHOLD 7.5f
#define PORT 60000 // todo mettila in params come renderer_port


int main() {


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
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);


	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// -----------------------------------------------------------------Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "../cpu_program/VertexShader.vertexshader", "../cpu_program/FragmentShader.fragmentshader" );
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint ModelMatrixID = glGetUniformLocation(programID, "Model"); 
    GLuint DensityID = glGetUniformLocation(programID, "voxelDensity");

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

	// Vertices colors based to Normal
    // Dati delle Normali (1 normale per ogni vertice, 6 facce x 2 triangoli x 3 vertici)
    static const GLfloat g_normal_buffer_data[] = {
        // Faccia Z- (Back)
        0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f,
        // Faccia Z+ (Front)
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        // Faccia X- (Left)
        -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        // Faccia X+ (Right)
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        // Faccia Y- (Bottom)
        0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        // Faccia Y+ (Top)
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_normal_buffer_data), g_normal_buffer_data, GL_STATIC_DRAW);

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

    

    // --------------------------SETUP TRANSLATION VECTORS--------------------
    // initializing rendering voxel space

    glm::vec3 ***voxelTranslationVectors = (glm::vec3***)malloc(NUM_VOXELS_X * sizeof(glm::vec3**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxelTranslationVectors[i] = (glm::vec3**) malloc(NUM_VOXELS_Y * sizeof(glm::vec3*));
        for(int j=0; j < NUM_VOXELS_Y; j++)
            voxelTranslationVectors[i][j] = (glm::vec3*) calloc(NUM_VOXELS_Z, sizeof(glm::vec3)); // allocates memory and writes all bytes to 0
    }

    for (int x = 0; x < NUM_VOXELS_X; ++x) {
        for (int y = 0; y < NUM_VOXELS_Y; ++y) {
            for (int z = 0; z < NUM_VOXELS_Z; ++z) {
                glm::vec3 translation(
                    (x * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_X,
                    (y * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Y,
                    (z * DIM_VOXEL + DIM_VOXEL / 2.0f) + MIN_Z
                );

                voxelTranslationVectors[x][y][z] = translation;
            }
        }
    }



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
    printf("Renderer listening on port %d...\n", PORT);

    client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
    if (client_fd < 0) {
        perror("Error accepting request");
        exit(1);
    }
    printf("Connected with Worker.\n\n");

    // -------------------------- RENDER LOOP --------------------
    float lastFrameTime = glfwGetTime();
    bool time_to_advance_frame;
    int num_points = 0;
    bool socket_closed = false;
    Voxel* active_voxels;
    int active_count = 0;

    do {
        //--------------------CHECK IF IT'S TIME TO UPDATE---------------------------
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;

        time_to_advance_frame = false;
        if (deltaTime >= FRAMEDURATION && !socket_closed) {
            time_to_advance_frame = true;
            lastFrameTime = currentTime;
        }

        if (time_to_advance_frame) {
            // Aggiorno il timer
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
        }


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

        // 2nd attribute buffer : normals
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer); // Usa normalbuffer!
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0
        );

        //---------------------render current voxels-------------------------------
        for (int i = 0; i < active_count; i++) {
            if (active_voxels[i].num_points > MIN_POINTS_IN_VOXEL_TO_RENDER) {
                //render it

                int x = active_voxels[i].x;
                int y = active_voxels[i].y;
                int z = active_voxels[i].z;

                // Calcolo posizione
                glm::vec3 t = voxelTranslationVectors[x][y][z];

                // Calcolo Model Matrix
                glm::mat4 Model1 = glm::translate(glm::mat4(1.0f), t);
                Model1 = glm::scale(Model1, glm::vec3(DIM_VOXEL / 2.0f));
                
                // Calcolo MVP
                glm::mat4 MVP1 = Projection * View * Model1;

                // INVIO UNIFORM ALLO SHADER
                // 1. MVP per la posizione dei vertici
                glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP1[0][0]);
                
                // 2. Model Matrix per il calcolo delle luci
                glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &Model1[0][0]);

                // 3. Densità per il colore (Heatmap)
                float density = (float)active_voxels[i].num_points / (float)MAX_DENSITY_THRESHOLD;
                glUniform1f(DensityID, density);

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

    for (int x = 0; x < NUM_VOXELS_X; ++x) {
        for (int y = 0; y < NUM_VOXELS_Y; ++y) {
            free(voxelTranslationVectors[x][y]);
        }
        free(voxelTranslationVectors[x]);
    }
    free(voxelTranslationVectors);

    free(active_voxels);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}