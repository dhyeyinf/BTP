/*
 * Auto-generated polynomial temperature estimation model
 * Model: T = f(R, T_env, H)
 * Polynomial degree: 4
 * 
 * Usage:
 *   float estimate_temperature(float R, float T_env, float H);
 * 
 * Generated from: temperature sensor calibration pipeline
 * Original model: GradientBoost
 * Test MAE: 3.093°C
 * Test R²: 0.9561
 * Generated on: 2025-11-16 14:44:55
 */

#ifndef SENSOR_POLY_MODEL_H
#define SENSOR_POLY_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_DEGREE 4
#define NUM_FEATURES 3
#define NUM_TERMS 34

#define FEATURE_R 0
#define FEATURE_T_ENV 1
#define FEATURE_H 2

static const float POLY_INTERCEPT = 2.6520193680e+03f;

typedef struct {
    unsigned char powers[NUM_FEATURES];
    float coefficient;
} PolyTerm;

int Rscale(int Rsens){
    if(Rsens > 10000)
        return Rsens -= 4000;
    return Rsens;
}

static const PolyTerm POLY_TERMS[NUM_TERMS] = {
    { { 1, 0, 0 }, -2.3182275633e+00f },  // R
    { { 0, 1, 0 }, -1.4881717225e+01f },  // T_env
    { { 0, 0, 1 }, 1.1906955198e+01f },  // H
    { { 2, 0, 0 }, 1.3469485163e-04f },  // R^2
    { { 1, 1, 0 }, 1.7559966997e-01f },  // R*T_env
    { { 1, 0, 1 }, 1.4140776325e-02f },  // R*H
    { { 0, 2, 0 }, 1.5267585063e+01f },  // T_env^2
    { { 0, 1, 1 }, -2.7523201692e+01f },  // T_env*H
    { { 0, 0, 2 }, 5.5002137951e+00f },  // H^2
    { { 3, 0, 0 }, -5.3536176242e-09f },  // R^3
    { { 2, 1, 0 }, -1.0889679628e-05f },  // R^2*T_env
    { { 2, 0, 1 }, 1.3357749454e-06f },  // R^2*H
    { { 1, 2, 0 }, -3.3063916536e-03f },  // R*T_env^2
    { { 1, 1, 1 }, -4.3636649107e-04f },  // R*T_env*H
    { { 1, 0, 2 }, -2.9194507255e-04f },  // R*H^2
    { { 0, 3, 0 }, -9.3072204296e-01f },  // T_env^3
    { { 0, 2, 1 }, 8.8973820087e-01f },  // T_env^2*H
    { { 0, 1, 2 }, 1.1342996999e-01f },  // T_env*H^2
    { { 0, 0, 3 }, -7.6915180445e-02f },  // H^3
    { { 4, 0, 0 }, 2.2987169727e-13f },  // R^4
    { { 3, 1, 0 }, 1.6401485842e-10f },  // R^3*T_env
    { { 3, 0, 1 }, -5.8867907872e-11f },  // R^3*H
    { { 2, 2, 0 }, 1.2313209326e-07f },  // R^2*T_env^2
    { { 2, 1, 1 }, 1.7332572647e-08f },  // R^2*T_env*H
    { { 2, 0, 2 }, -4.5037227123e-09f },  // R^2*H^2
    { { 1, 3, 0 }, 1.7349454372e-05f },  // R*T_env^3
    { { 1, 2, 1 }, 2.9413736322e-06f },  // R*T_env^2*H
    { { 1, 1, 2 }, 3.8923167876e-07f },  // R*T_env*H^2
    { { 1, 0, 3 }, 1.8962776464e-06f },  // R*H^3
    { { 0, 4, 0 }, 1.3712276546e-02f },  // T_env^4
    { { 0, 3, 1 }, -7.8674858799e-03f },  // T_env^3*H
    { { 0, 2, 2 }, -2.7060919398e-03f },  // T_env^2*H^2
    { { 0, 1, 3 }, 1.2119715242e-04f },  // T_env*H^3
    { { 0, 0, 4 }, 3.0109217106e-04f },  // H^4
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

#endif // SENSOR_POLY_MODEL_H
