
#ifndef NESTED_FUNC_H
#define NESTED_FUNC_H

#include "nested_func_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
} point_t;

typedef struct {
    point_t center;
    point_t points[3];
    float weights[3];
} cluster_t;

// Input arguments struct
typedef struct {
    float scalar;
    float arr[3];
    cluster_t clusters[2];
    // empty_struct was empty, no fields generated
} nested_func_arg_t;

// Output results struct
typedef struct {
    float z;
} nested_func_res_t;

// Workspace struct
typedef struct {
    long int iw[nested_func_SZ_IW];
    float w[nested_func_SZ_W];
} nested_func_work_t;

// Runtime API
int nested_func_init(nested_func_arg_t* arg, nested_func_res_t* res, nested_func_work_t* work);
int nested_func_step(nested_func_arg_t* arg, nested_func_res_t* res, nested_func_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // NESTED_FUNC_H