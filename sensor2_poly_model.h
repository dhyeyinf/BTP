/*
 * Auto-generated polynomial temperature estimation model
 * Model: T = f(R, T_env, H)
 * Polynomial degree: 4
 */

#ifndef SENSOR2_POLY_MODEL_H
#define SENSOR2_POLY_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_DEGRE 4
#define NUM_FEATURESS 3
#define NUM_TERMSS 34

#define FEATURE_ 0
#define FEATURE_T_EN 1
#define FEATURE_H_ENV 2

static const float POLY_INTERCEPTS = -1.1546523928e+04f;

#include <stdlib.h>
#include <stdint.h>

/* KEEPING YOUR STRUCT NAME */
typedef struct {
    unsigned char powers[NUM_FEATURESS];
    float coefficient;
} PolyTerms;

/* KEEPING YOUR FUNCTION */
int Rscale(int Rsens){
    if(Rsens > 10000)
        return Rsens -= 7000;
    return Rsens;
}

/* KEEPING YOUR ARRAY NAME AND SIZE */
static const PolyTerms POLY_TERMSS[NUM_TERMSS] = {
    { { 1, 0, 0 }, 6.9432163936e-05f },
    { { 0, 1, 0 }, 2.6023903669e-06f },
    { { 0, 0, 1 }, -1.8668915807e-06f },
    { { 2, 0, 0 }, 3.2239633383e-02f },
    { { 1, 1, 0 }, 1.2235791148e-03f },
    { { 1, 0, 1 }, 1.2237622572e-03f },
    { { 0, 2, 0 }, 4.3567075096e-05f },
    { { 0, 1, 1 }, -4.2243134102e-05f },
    { { 0, 0, 2 }, -7.4659918545e-05f },
    { { 3, 0, 0 }, 8.8378068338e-08f },
    { { 2, 1, 0 }, -1.8702025544e-03f },
    { { 2, 0, 1 }, -4.3536190934e-04f },
    { { 1, 2, 0 }, 1.6283120120e-02f },
    { { 1, 1, 1 }, 1.7783423045e-02f },
    { { 1, 0, 2 }, -3.4966161715e-03f },
    { { 0, 3, 0 }, 2.1139218611e-03f },
    { { 0, 2, 1 }, -8.3034945018e-04f },
    { { 0, 1, 2 }, -1.9884278540e-03f },
    { { 0, 0, 3 }, -6.3891560843e-04f },
    { { 4, 0, 0 }, 9.2709703160e-14f },
    { { 3, 1, 0 }, -1.9560443985e-09f },
    { { 3, 0, 1 }, -8.0195300113e-10f },
    { { 2, 2, 0 }, 2.3360666545e-05f },
    { { 2, 1, 1 }, 1.5079786294e-05f },
    { { 2, 0, 2 }, 1.0682299436e-06f },
    { { 1, 3, 0 }, -2.8444679476e-03f },
    { { 1, 2, 1 }, 2.5915979027e-03f },
    { { 1, 1, 2 }, -1.3632864949e-03f },
    { { 1, 0, 3 }, 2.0374501964e-04f },
    { { 0, 4, 0 }, 8.9460019818e-02f },
    { { 0, 3, 1 }, -4.0748228492e-03f },
    { { 0, 2, 2 }, -7.2770541119e-02f },
    { { 0, 1, 3 }, 4.2296356088e-02f },
    { { 0, 0, 4 }, -6.2101956347e-03f }
};

static inline void random_init(uint32_t seed)
{
    srand(seed);
}

static inline float Tscale(float x, float min_val, float max_val)
{
    float r = (float)rand() / (float)RAND_MAX;
    return min_val + r * (max_val - min_val);
}

/* KEEPING YOUR FUNCTION NAME AND VARIABLES */
static inline float estimate_temperatures(float R, float T_env, float H) {

    float features[NUM_FEATURESS] = { R, T_env, H };
    float result = POLY_INTERCEPTS;

    for (int i = 0; i < NUM_TERMSS; i++) {

        float term_value = POLY_TERMSS[i].coefficient;

        for (int j = 0; j < NUM_FEATURESS; j++) {
            unsigned char power = POLY_TERMSS[i].powers[j];

            for (unsigned char p = 0; p < power; p++) {
                term_value *= features[j];
            }
        }

        result += term_value;
    }

    return result;
}

#ifdef __cplusplus
}
#endif

#endif // SENSOR2_POLY_MODEL_H
