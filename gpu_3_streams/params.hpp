#define FIELDS_PER_POINT 4

#define MAX_X 50 //34
#define MAX_Y 50 // 22
#define MAX_Z 10 // 3
#define MIN_X -50 //-34
#define MIN_Y -50 // -22
#define MIN_Z -10 // -3

#define MIN_POINTS_IN_VOXEL_TO_RENDER 0
#define PORT 53456

#define DIM_VOXEL 0.1

#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))

#define NUM_TOT_VOXELS NUM_VOXELS_X * NUM_VOXELS_Y * NUM_VOXELS_Z

#define DIRNAME "../new_dataset"

#define FRAMEDURATION 0.1f // 10 FPS

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