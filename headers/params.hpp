#define FIELDS_PER_POINT 4
#define MAX_POINTS_PER_BUFFER 131100
#define NUM_BUFFERS 1

#define MAX_X 50 //34
#define MAX_Y 50 // 22
#define MAX_Z 10 // 3
#define MIN_X -50 //-34
#define MIN_Y -50 // -22
#define MIN_Z -10 // -3

#define MIN_POINTS_IN_VOXEL_TO_RENDER 0
#define MAX_DENSITY_THRESHOLD 7.5f

#define WORKER_PORT 53456
#define RENDERER_PORT 60000

#define DIM_VOXEL 0.1f

#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))

#define NUM_TOT_VOXELS NUM_VOXELS_X * NUM_VOXELS_Y * NUM_VOXELS_Z

#define DIRNAME "../new_dataset"

#define FRAMEDURATION 1.0f // 10 FPS

#define WINDOWNAME "GPU Voxelization"

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

typedef struct {
  int x;
  int y;
  int z;
  int num_points;
} Voxel;

struct CallbackData {
    int socket_fd;       // La tua "data[0]"
    void* buffer_ptr;    // Il puntatore al buffer da inviare
    size_t data_size;    // La dimensione in byte (NUM_TOT_VOXELS * sizeof(int))
    int buffer_id;       // Solo per fare print di debug corretti (opzionale)
};