
#ifndef DICT_FUNC_H
#define DICT_FUNC_H

#include "dict_func_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float lr;
    float momentum;
} config_t;

typedef struct {
    // none_res was empty, no fields generated
    float result;
} output_t;

// Input arguments struct
typedef struct {
    config_t config;
    float bounds[2];
    // empty_dict was empty, no fields generated
    // empty_list was empty, no fields generated
    float single_tuple[1];
    // none_arg was empty, no fields generated
} dict_func_arg_t;

// Output results struct
typedef struct {
    output_t output;
} dict_func_res_t;

// Workspace struct
typedef struct {
    long int iw[dict_func_SZ_IW];
    float w[dict_func_SZ_W];
} dict_func_work_t;

// Runtime API
int dict_func_init(dict_func_arg_t* arg, dict_func_res_t* res, dict_func_work_t* work);
int dict_func_step(dict_func_arg_t* arg, dict_func_res_t* res, dict_func_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // DICT_FUNC_H