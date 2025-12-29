#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
using namespace glm;
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

#include "common/shader.hpp"

#define FIELDS_PER_POINT 4

#define MAX_X 34
#define MIN_X -34

#define MAX_Y 22
#define MIN_Y -22

#define MAX_Z 3
#define MIN_Z -3

#define DIM_VOXEL 0.1

#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))

#define NUM_TOT_VOXELS NUM_VOXELS_X * NUM_VOXELS_Y * NUM_VOXELS_Z

#define DIRNAME "../new_dataset"

#define FRAMEDURATION 1.0f // 10 FPS

#define true 1
#define false 0



typedef struct {
  float x;
  float y;
  float z;
} Point;

typedef struct {
  int i;
  int j;
  int k;
} VoxelIndices;



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

VoxelIndices calculate_voxel_indices(float* point) {
    VoxelIndices result;
    result.i = (int)floor((point[0] - MIN_X) / DIM_VOXEL);
    result.j = (int)floor((point[1] - MIN_Y) / DIM_VOXEL);
    result.k = (int)floor((point[2] - MIN_Z) / DIM_VOXEL);
    return result;
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
	window = glfwCreateWindow( 1024, 768, "Tutorial 02 - Cube", NULL, NULL);
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
	static const GLfloat g_color_buffer_data[] = {
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f
	};

    GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);


	// Modify camera position
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), float(width)/float(height), 0.1f, 100.0f);
	glm::mat4 View       = glm::lookAt(
								glm::vec3(0, 5, 20), // Camera position
								glm::vec3(0,0,0), // Look at origin
								glm::vec3(0,1,0)  // Head is up
						);
	glm::mat4 Model      = glm::mat4(1.0f);
	glm::mat4 MVP        = Projection * View * Model;


    // --------------------------SETUP TRANSLATION VECTORS--------------------
    // inizializzo spazio di rendering voxel
    // vettore lineare di traslazioni (X * Y * Z)

    glm::vec3 ***voxelTranslations = (glm::vec3***)malloc(NUM_VOXELS_X * sizeof(glm::vec3**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxelTranslations[i] = (glm::vec3**) malloc(NUM_VOXELS_Y * sizeof(glm::vec3*));
        for(int j=0; j < NUM_VOXELS_Y; j++)
            voxelTranslations[i][j] = (glm::vec3*) calloc(NUM_VOXELS_Z, sizeof(glm::vec3)); // allocates memory and writes all bytes to 0
    }

    // popolamento array
    for (int x = 0; x < NUM_VOXELS_X; ++x) {
        for (int y = 0; y < NUM_VOXELS_Y; ++y) {
            for (int z = 0; z < NUM_VOXELS_Z; ++z) {
                // centro la griglia nell'angolo in basso a sinistra corrispondente al primo voxel
                glm::vec3 translation(
                    x * DIM_VOXEL + DIM_VOXEL / 2.0f,
                    y * DIM_VOXEL + DIM_VOXEL / 2.0f,
                    z * DIM_VOXEL + DIM_VOXEL / 2.0f
                );

                voxelTranslations[x][y][z] = translation;
            }
        }
    }

    // per applicare la traslazione : gl_Position = projection * view * vec4(worldPos, 1.0);

    // --------------------------SETUP VOXELS MATRIX--------------------------
    int ***voxels = (int***) malloc(NUM_VOXELS_X * sizeof(int**));
    for(int i=0; i<NUM_VOXELS_X; i++) {
        voxels[i] = (int**) malloc(NUM_VOXELS_Y * sizeof(int*));
        for(int j=0; j<NUM_VOXELS_Y; j++)
            voxels[i][j] = (int*) calloc(NUM_VOXELS_Z, sizeof(int)); // allocates memory and writes all bytes to 0
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
    
    // FOR EACH FRAME VOXELIZE THE POINT CLOUD
    char path_to_current_frame[512];
    struct dirent* entry;
    FILE* current_frame;
    char* fname;
    float point[FIELDS_PER_POINT];
    int num_frame = 0, num_points;
    float lastFrameTime = glfwGetTime();

    do {
        //--------------------CHECK IF IT'S TIME TO UPDATE---------------------------
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;

        bool time_to_advance_frame = false;
        if (deltaTime >= FRAMEDURATION) {
            time_to_advance_frame = true;
            lastFrameTime = currentTime;
        }
        // ----------------------------UPDATE VOXEL DATA----------------------------
        if (time_to_advance_frame) {
            //get the new dir entry
            if (num_frame >= file_count) {
                break;
            }
            fname = file_names[num_frame];
            sprintf(path_to_current_frame, "%s/%s", DIRNAME, fname);
            // caricamento dati in memoria
            current_frame = fopen(path_to_current_frame, "rb");
            if (current_frame == NULL) {
                perror("Errore apertura file input");
                continue;
            }
            //num_points = calculate_num_points(current_frame);
            // Reinizializza voxels a tutti zeri
            for(int i = 0; i < NUM_VOXELS_X; i++) {
                for(int j = 0; j < NUM_VOXELS_Y; j++) {
                    memset(voxels[i][j], 0, NUM_VOXELS_Z * sizeof(int));
                }
            }
            // Lettura
            while (fread(point, sizeof(float), FIELDS_PER_POINT, current_frame) == FIELDS_PER_POINT) {
                // trova il voxel in cui è
                VoxelIndices curr_voxel_indices = calculate_voxel_indices(point);


                if(curr_voxel_indices.i < 0 || curr_voxel_indices.i >= NUM_VOXELS_X ||
                    curr_voxel_indices.j < 0 || curr_voxel_indices.j >= NUM_VOXELS_Y ||
                    curr_voxel_indices.k < 0 || curr_voxel_indices.k >= NUM_VOXELS_Z) {
                        // punto fuori dai limiti
                        continue;
                }

                voxels[curr_voxel_indices.i][curr_voxel_indices.j][curr_voxel_indices.k]++;
            }
            fclose(current_frame);
            printf("FINITO FILE %s\n", entry->d_name);
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

        // 2nd attribute buffer : colors
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0
        );

        //---------------------render current voxels-------------------------------
        for (int i = 0; i < NUM_VOXELS_X; i++) {
            for (int j = 0; j < NUM_VOXELS_Y; j++) {
                for (int k = 0; k < NUM_VOXELS_Z; k++) {
                    if (voxels[i][j][k] > 0) {
                        //render it
                        glm::mat4 Model1 = glm::translate(glm::mat4(1.0f), voxelTranslations[i][j][k]);
                        glm::mat4 MVP1 = Projection * View * Model1;
                        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP1[0][0]);
                        glDrawArrays(GL_TRIANGLES, 0, 12*3);
                    }
                }
            }
        }

        // --- CUBE EXAMPLE: LEFT TRANSLATED ---
        // glm::mat4 Model1 = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 1.0f, 0.0f));
        // glm::mat4 MVP1 = Projection * View * Model1;
        // glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP1[0][0]);
        // glDrawArrays(GL_TRIANGLES, 0, 12*3);
        // END OF EXAMPLE---------------------------------

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

	return 0;
}
