typedef struct {
  float x;
  float y;
  float z;
} Point;


#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>

#define FIELDS_PER_POINT 4

#define DIM_VOXEL 0.1
#define NUM_VOXELS_X ((int)((MAX_X - MIN_X)/DIM_VOXEL))
#define NUM_VOXELS_Y ((int)((MAX_Y - MIN_Y)/DIM_VOXEL))
#define NUM_VOXELS_Z ((int)((MAX_Z - MIN_Z)/DIM_VOXEL))

#define NUM_TOT_VOXELS NUM_VOXELS_X * NUM_VOXELS_Y * NUM_VOXELS_Z

#define MAX_X 300 //34
#define MAX_Y 200 // 22
#define MAX_Z 100 // 3
#define MIN_X -300 //-34
#define MIN_Y -200 // -22
#define MIN_Z -100 // -3

#define DIR_NAME "../new_dataset"

#define PORT 53456