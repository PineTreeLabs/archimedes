
#ifndef COMBINE_FUNC_H
#define COMBINE_FUNC_H

#include "combine_func_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float data[5];
    // more_data was empty, no fields generated
} container_5_t;

typedef struct {
    float data[2];
    // more_data was empty, no fields generated
} container_2_t;

typedef struct {
    float data[3];
    float more_data[2];
} container_3_2_t;

typedef struct {
    float data[2];
    float more_data[3];
} container_2_3_t;

typedef struct {
    float data[3];
    // more_data was empty, no fields generated
} container_3_t;

// Input arguments struct
typedef struct {
    container_3_t c1;
    container_2_t c2;
} combine_func_arg_t;

// Output results struct
typedef struct {
    container_5_t combined;
    container_2_t c2_out;
    container_3_2_t extra1;
    container_2_3_t extra2;
} combine_func_res_t;

// Workspace struct
typedef struct {
    long int iw[combine_func_SZ_IW];
    float w[combine_func_SZ_W];
} combine_func_work_t;

// Runtime API
int combine_func_init(combine_func_arg_t* arg, combine_func_res_t* res, combine_func_work_t* work);
int combine_func_step(combine_func_arg_t* arg, combine_func_res_t* res, combine_func_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // COMBINE_FUNC_H