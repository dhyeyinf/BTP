/*
 * Auto-generated polynomial temperature estimation model
 * Model: T = f(R, T_env, H)
 * Polynomial degree: 4
 * 
 * Usage:
 *   float estimate_temperature(float R, float T_env, float H);
 * 
 * Generated from: sensor3-v2.ipynb
 */

#ifndef SENSOR3_POLY_MODEL_H
#define SENSOR3_POLY_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_DEGREE 4
#define NUM_FEATURES 3
#define NUM_TERMS 34

#define FEATURE_R 0
#define FEATURE_T_ENV 1
#define FEATURE_H 2

static const float POLY_INTERCEPT = -1.0724699638e+04f;

typedef struct {
    unsigned char powers[NUM_FEATURES];
    float coefficient;
} PolyTerm;
int Rscale(int Rsens){
    if(Rsens>10000)
        return Rsens -=7000;
}
static const PolyTerm POLY_TERMS[NUM_TERMS] = {
    { { 1, 0, 0 }, -3.1521278851e-01f },  // R
    { { 0, 1, 0 }, 1.0696859891e-03f },  // T_env
    { { 0, 0, 1 }, -4.7341782325e-05f },  // H
    { { 2, 0, 0 }, -2.6647300527e-03f },  // R*R
    { { 1, 1, 0 }, 1.1998797924e+00f },  // R*T_env
    { { 1, 0, 1 }, -2.6585139354e-02f },  // R*H
    { { 0, 2, 0 }, 2.4552227983e-02f },  // T_env*T_env
    { { 0, 1, 1 }, 3.3668061718e-02f },  // T_env*H
    { { 0, 0, 2 }, -3.9103807378e-03f },  // H*H
    { { 3, 0, 0 }, -4.0641243044e-08f },  // R*R*R
    { { 2, 1, 0 }, 1.9211342768e-04f },  // R*R*T_env
    { { 2, 0, 1 }, 1.8920160053e-05f },  // R*R*H
    { { 1, 2, 0 }, -5.9920862917e-02f },  // R*T_env*T_env
    { { 1, 1, 1 }, -2.1558906347e-02f },  // R*T_env*H
    { { 1, 0, 2 }, 3.0396453085e-03f },  // R*H*H
    { { 0, 3, 0 }, 2.5397954859e-01f },  // T_env*T_env*T_env
    { { 0, 2, 1 }, 4.7738264102e-01f },  // T_env*T_env*H
    { { 0, 1, 2 }, 6.5354896145e-01f },  // T_env*H*H
    { { 0, 0, 3 }, -1.5244951190e-01f },  // H*H*H
    { { 4, 0, 0 }, -4.2880156571e-12f },  // R*R*R*R
    { { 3, 1, 0 }, 2.2411756585e-09f },  // R*R*R*T_env
    { { 3, 0, 1 }, 1.1616956278e-09f },  // R*R*R*H
    { { 2, 2, 0 }, -7.1628738818e-06f },  // R*R*T_env*T_env
    { { 2, 1, 1 }, 1.8477050311e-06f },  // R*R*T_env*H
    { { 2, 0, 2 }, -6.1771161043e-07f },  // R*R*H*H
    { { 1, 3, 0 }, 2.5391876155e-03f },  // R*T_env*T_env*T_env
    { { 1, 2, 1 }, -7.0541694602e-04f },  // R*T_env*T_env*H
    { { 1, 1, 2 }, 2.4555455107e-04f },  // R*T_env*H*H
    { { 1, 0, 3 }, -1.0106297371e-05f },  // R*H*H*H
    { { 0, 4, 0 }, -8.0637554820e-02f },  // T_env*T_env*T_env*T_env
    { { 0, 3, 1 }, -3.1431700690e-02f },  // T_env*T_env*T_env*H
    { { 0, 2, 2 }, 4.0179782869e-02f },  // T_env*T_env*H*H
    { { 0, 1, 3 }, -1.8473938559e-02f },  // T_env*H*H*H
    { { 0, 0, 4 }, 2.3025075844e-03f }  // H*H*H*H
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

    return result;
}

#ifdef __cplusplus
}
#endif

#endif // SENSOR3_POLY_MODEL_H
