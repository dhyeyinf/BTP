/*
 * Auto-generated polynomial temperature estimation model
 * Model: T = f(R, T_env, H)
 * Polynomial degree: 3
 * 
 * Usage:
 *   float estimate_temperature_v2(float R, float T_env, float H);
 * 
 * Generated from: predict_temperature_v2()
 */

#ifndef SENSOR3_POLY_MODEL_V2_H
#define SENSOR3_POLY_MODEL_V2_H

#ifdef __cplusplus
extern "C" {
#endif
#define mean 4712.4160
#define stddev 296.4665
#define meanT 25
#define stddevT 23
#define POLY_DEGREE 3
#define NUM_FEATURES 3
#define NUM_TERMS 20
#define FEATURE_R 0
#define FEATURE_T_ENV 1
#define FEATURE_H 2

static const float POLY_INTERCEPT = -14641.5f;
#include "sensor2_poly_model.h"
#undef mean
#undef stddev
#undef meanT
#undef stddevT
typedef struct {
    unsigned char powers[NUM_FEATURES];
    float coefficient;
} PolyTerm;


float RScale(int x, float mean, float stddev)
{
    return (x - mean) / stddev;
}

float TScale(int x, float meanT, float stddevT)
{
    return (x - meanT) / stddevT;
}

static const PolyTerm POLY_TERMS[NUM_TERMS] = {
    { { 1, 0, 0 }, -0.121648f },
    { { 0, 1, 0 }, 0.00294532f },
    { { 0, 0, 1 }, -0.00374173f },
    { { 2, 0, 0 }, -0.00207366f },
    { { 1, 1, 0 }, 0.589058f },
    { { 1, 0, 1 }, 0.0966738f },
    { { 0, 2, 0 }, 0.107848f },
    { { 0, 1, 1 }, 0.0310788f },
    { { 0, 0, 2 }, -0.2223f },
    { { 3, 0, 0 }, -9.39617e-09f },
    { { 2, 1, 0 }, 5.52031e-05f },
    { { 2, 0, 1 }, 1.51367e-05f },
    { { 1, 2, 0 }, -0.0788267f },
    { { 1, 1, 1 }, 0.0396814f },
    { { 1, 0, 2 }, -0.00909191f },
    { { 0, 3, 0 }, 2.58346f },
    { { 0, 2, 1 }, 1.98747f },
    { { 0, 1, 2 }, -2.1669f },
    { { 0, 0, 3 }, 0.467417f }
};

static inline float estimate_temperature(float R, float T_env, float H) {
    float features[NUM_FEATURES] = {R, T_env, H};
    float result = POLY_INTERCEPT;

    for (int i = 0; i < NUM_TERMS; i++) {
        float term_value = POLY_TERMS[i].coefficient;
        for (int j = 0; j < NUM_FEATURES; j++) {
            unsigned char power = POLY_TERMS[i].powers[j];
            for (unsigned char p = 0; p < power; p++) {
                term_value *= features[j];
            }
        }
        result += term_value;
    }
    result= Tscale(result, 23, 25);

    return result;
}



#ifdef __cplusplus
}
#endif

#endif // SENSOR3_POLY_MODEL_V2_H
