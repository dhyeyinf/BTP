/*******************************************************************************
 * FabricTempSensor.h
 * 
 * Advanced Textile-Based Temperature Sensor Library with AI-Driven Analytics
 * 
 * Description:
 *   A comprehensive embedded machine learning library for textile-based 
 *   temperature sensing systems. This library implements a polynomial regression
 *   model trained on real sensor data to predict fabric surface temperature
 *   based on resistance measurements, environmental temperature, and humidity.
 * 
 * Features:
 *   - High-accuracy polynomial regression model (Degree 4)
 *   - Multi-feature input processing (Resistance, Temperature, Humidity)
 *   - Automatic resistance scaling and calibration
 *   - Statistical analysis functions (MAE, RMSE, R²)
 *   - Data validation and error handling
 *   - Memory-efficient implementation for embedded systems
 *   - Diagnostic and debugging utilities
 *   - Model performance metrics and versioning
 * 
 * Model Specifications:
 *   - Algorithm: Gradient Boosting → Polynomial Approximation
 *   - Polynomial Degree: 4
 *   - Input Features: 3 (R, T_env, H)
 *   - Total Terms: 34 polynomial terms
 *   - Test MAE: 3.093°C
 *   - Test R²: 0.9561
 *   - Training Dataset: 1500+ calibrated measurements
 * 
 * Hardware Requirements:
 *   - ESP32 or compatible microcontroller
 *   - Fabric resistance sensor (thermistor-based textile)
 *   - HTU21D environmental sensor (or equivalent)
 *   - MAX6675 thermocouple (for validation)
 * 
 * Usage Example:
 *   #include "FabricTempSensor.h"
 *   
 *   FabricTempSensor sensor;
 *   sensor.begin();
 *   
 *   float resistance = 8500.0;
 *   float env_temp = 25.5;
 *   float humidity = 45.0;
 *   
 *   float predicted_temp = sensor.predictTemperature(resistance, env_temp, humidity);
 *   
 *   if (sensor.isValidPrediction(predicted_temp)) {
 *       Serial.printf("Fabric Temperature: %.2f°C\n", predicted_temp);
 *   }
 * 

 ******************************************************************************/

#ifndef FABRIC_TEMP_SENSOR_H
#define FABRIC_TEMP_SENSOR_H

#include <Arduino.h>
#include <stdint.h>
#include <math.h>

/*******************************************************************************
 * LIBRARY VERSION INFORMATION
 ******************************************************************************/
#define FABRIC_TEMP_SENSOR_VERSION_MAJOR 2
#define FABRIC_TEMP_SENSOR_VERSION_MINOR 0
#define FABRIC_TEMP_SENSOR_VERSION_PATCH 1
#define FABRIC_TEMP_SENSOR_VERSION "2.0.1"

/*******************************************************************************
 * MODEL CONFIGURATION
 ******************************************************************************/
#define POLY_DEGREE 4
#define NUM_FEATURES 3
#define NUM_TERMS 34

// Feature indices for clarity
#define FEATURE_RESISTANCE 0
#define FEATURE_ENV_TEMP 1
#define FEATURE_HUMIDITY 2

// Model metadata
#define MODEL_NAME "GradientBoost-Polynomial-v2"
#define MODEL_TRAINING_DATE "2025-11-16"
#define MODEL_TEST_MAE 3.093f
#define MODEL_TEST_R2 0.9561f
#define MODEL_TRAINING_SAMPLES 1500

/*******************************************************************************
 * SENSOR CONFIGURATION PARAMETERS
 ******************************************************************************/
// Resistance sensor specifications
#define DEFAULT_RESISTANCE_MIN 1000.0f    // Minimum expected resistance (Ω)
#define DEFAULT_RESISTANCE_MAX 20000.0f   // Maximum expected resistance (Ω)
#define RESISTANCE_SCALE_THRESHOLD 10000.0f
#define RESISTANCE_SCALE_OFFSET 4000.0f

// Environmental sensor ranges
#define ENV_TEMP_MIN -10.0f               // Minimum environmental temp (°C)
#define ENV_TEMP_MAX 60.0f                // Maximum environmental temp (°C)
#define HUMIDITY_MIN 0.0f                 // Minimum humidity (%)
#define HUMIDITY_MAX 100.0f               // Maximum humidity (%)

// Temperature prediction bounds
#define PREDICTED_TEMP_MIN -20.0f         // Minimum valid prediction (°C)
#define PREDICTED_TEMP_MAX 150.0f         // Maximum valid prediction (°C)

// Error codes
#define ERROR_NONE 0
#define ERROR_INVALID_RESISTANCE -1
#define ERROR_INVALID_ENV_TEMP -2
#define ERROR_INVALID_HUMIDITY -3
#define ERROR_PREDICTION_OUT_OF_BOUNDS -4
#define ERROR_SENSOR_NOT_INITIALIZED -5

/*******************************************************************************
 * POLYNOMIAL TERM STRUCTURE
 ******************************************************************************/
typedef struct {
    uint8_t powers[NUM_FEATURES];  // Exponents for each feature
    float coefficient;              // Polynomial coefficient
} PolyTerm;

/*******************************************************************************
 * SENSOR STATISTICS STRUCTURE
 ******************************************************************************/
typedef struct {
    float mean;
    float stddev;
    float min;
    float max;
    uint32_t count;
    float sum;
    float sum_squared;
} SensorStats;

/*******************************************************************************
 * PREDICTION RESULT STRUCTURE
 ******************************************************************************/
typedef struct {
    float temperature;              // Predicted temperature (°C)
    float confidence;               // Confidence score (0-1)
    int8_t error_code;             // Error code (0 = success)
    uint32_t computation_time_us;  // Computation time (microseconds)
    bool is_valid;                 // Overall validity flag
} PredictionResult;

/*******************************************************************************
 * MODEL COEFFICIENTS - POLYNOMIAL TERMS
 * 
 * This array contains 34 polynomial terms representing the trained ML model.
 * Each term consists of:
 *   - powers[3]: Exponents for [R, T_env, H]
 *   - coefficient: Multiplicative coefficient
 * 
 * The polynomial equation is:
 *   T_fabric = INTERCEPT + Σ(coefficient × R^a × T_env^b × H^c)
 *   
 * Where a, b, c are the powers for each term.
 ******************************************************************************/
static const float POLY_INTERCEPT = 2.6520193680e+03f;

static const PolyTerm POLY_TERMS[NUM_TERMS] = {
    // First-order terms (linear)
    { { 1, 0, 0 }, -2.3182275633e+00f },  // R
    { { 0, 1, 0 }, -1.4881717225e+01f },  // T_env
    { { 0, 0, 1 }, 1.1906955198e+01f },   // H
    
    // Second-order terms (quadratic)
    { { 2, 0, 0 }, 1.3469485163e-04f },   // R²
    { { 1, 1, 0 }, 1.7559966997e-01f },   // R × T_env
    { { 1, 0, 1 }, 1.4140776325e-02f },   // R × H
    { { 0, 2, 0 }, 1.5267585063e+01f },   // T_env²
    { { 0, 1, 1 }, -2.7523201692e+01f },  // T_env × H
    { { 0, 0, 2 }, 5.5002137951e+00f },   // H²
    
    // Third-order terms (cubic)
    { { 3, 0, 0 }, -5.3536176242e-09f },  // R³
    { { 2, 1, 0 }, -1.0889679628e-05f },  // R² × T_env
    { { 2, 0, 1 }, 1.3357749454e-06f },   // R² × H
    { { 1, 2, 0 }, -3.3063916536e-03f },  // R × T_env²
    { { 1, 1, 1 }, -4.3636649107e-04f },  // R × T_env × H
    { { 1, 0, 2 }, -2.9194507255e-04f },  // R × H²
    { { 0, 3, 0 }, -9.3072204296e-01f },  // T_env³
    { { 0, 2, 1 }, 8.8973820087e-01f },   // T_env² × H
    { { 0, 1, 2 }, 1.1342996999e-01f },   // T_env × H²
    { { 0, 0, 3 }, -7.6915180445e-02f },  // H³
    
    // Fourth-order terms (quartic)
    { { 4, 0, 0 }, 2.2987169727e-13f },   // R⁴
    { { 3, 1, 0 }, 1.6401485842e-10f },   // R³ × T_env
    { { 3, 0, 1 }, -5.8867907872e-11f },  // R³ × H
    { { 2, 2, 0 }, 1.2313209326e-07f },   // R² × T_env²
    { { 2, 1, 1 }, 1.7332572647e-08f },   // R² × T_env × H
    { { 2, 0, 2 }, -4.5037227123e-09f },  // R² × H²
    { { 1, 3, 0 }, 1.7349454372e-05f },   // R × T_env³
    { { 1, 2, 1 }, 2.9413736322e-06f },   // R × T_env² × H
    { { 1, 1, 2 }, 3.8923167876e-07f },   // R × T_env × H²
    { { 1, 0, 3 }, 1.8962776464e-06f },   // R × H³
    { { 0, 4, 0 }, 1.3712276546e-02f },   // T_env⁴
    { { 0, 3, 1 }, -7.8674858799e-03f },  // T_env³ × H
    { { 0, 2, 2 }, -2.7060919398e-03f },  // T_env² × H²
    { { 0, 1, 3 }, 1.2119715242e-04f },   // T_env × H³
    { { 0, 0, 4 }, 3.0109217106e-04f },   // H⁴
};

/*******************************************************************************
 * MAIN LIBRARY CLASS
 ******************************************************************************/
class FabricTempSensor {
private:
    bool _initialized;
    uint32_t _prediction_count;
    SensorStats _resistance_stats;
    SensorStats _env_temp_stats;
    SensorStats _humidity_stats;
    SensorStats _prediction_stats;
    
    // Private helper methods
    void updateStats(SensorStats* stats, float value);
    void resetStats(SensorStats* stats);
    float calculatePower(float base, uint8_t exponent);
    
public:
    /***************************************************************************
     * Constructor
     ***************************************************************************/
    FabricTempSensor();
    
    /***************************************************************************
     * Initialization
     * 
     * Initializes the sensor library and resets all statistics.
     * Must be called before using prediction functions.
     ***************************************************************************/
    void begin();
    
    /***************************************************************************
     * Resistance Scaling Function
     * 
     * Applies calibration scaling to raw resistance measurements.
     * This compensates for known sensor non-linearities.
     * 
     * Parameters:
     *   resistance - Raw resistance value (Ω)
     * 
     * Returns:
     *   Calibrated resistance value (Ω)
     ***************************************************************************/
    float scaleResistance(float resistance);
    
    /***************************************************************************
     * Core Temperature Prediction Function
     * 
     * Predicts fabric temperature using the polynomial regression model.
     * 
     * Parameters:
     *   resistance - Fabric sensor resistance (Ω)
     *   env_temp   - Environmental temperature (°C)
     *   humidity   - Relative humidity (%)
     * 
     * Returns:
     *   Predicted fabric temperature (°C)
     ***************************************************************************/
    float predictTemperature(float resistance, float env_temp, float humidity);
    
    /***************************************************************************
     * Advanced Prediction with Result Structure
     * 
     * Performs prediction and returns comprehensive result information
     * including error codes, confidence, and computation time.
     * 
     * Parameters:
     *   resistance - Fabric sensor resistance (Ω)
     *   env_temp   - Environmental temperature (°C)
     *   humidity   - Relative humidity (%)
     * 
     * Returns:
     *   PredictionResult structure with detailed information
     ***************************************************************************/
    PredictionResult predictTemperatureAdvanced(float resistance, float env_temp, float humidity);
    
    /***************************************************************************
     * Input Validation Functions
     ***************************************************************************/
    bool validateResistance(float resistance);
    bool validateEnvTemp(float env_temp);
    bool validateHumidity(float humidity);
    bool isValidPrediction(float temperature);
    
    /***************************************************************************
     * Statistical Analysis Functions
     ***************************************************************************/
    SensorStats getResistanceStats();
    SensorStats getEnvTempStats();
    SensorStats getHumidityStats();
    SensorStats getPredictionStats();
    uint32_t getPredictionCount();
    void resetStatistics();
    
    /***************************************************************************
     * Model Information Functions
     ***************************************************************************/
    const char* getModelName();
    const char* getModelVersion();
    const char* getLibraryVersion();
    float getModelMAE();
    float getModelR2();
    int getModelTermCount();
    
    /***************************************************************************
     * Diagnostic Functions
     ***************************************************************************/
    void printModelInfo();
    void printStatistics();
    void printPredictionResult(PredictionResult result);
    
    /***************************************************************************
     * Calculate Mean Absolute Error (MAE)
     * 
     * Computes MAE between predicted and actual temperatures.
     * Useful for model validation and performance monitoring.
     ***************************************************************************/
    float calculateMAE(float* predicted, float* actual, int count);
    
    /***************************************************************************
     * Calculate Root Mean Squared Error (RMSE)
     ***************************************************************************/
    float calculateRMSE(float* predicted, float* actual, int count);
    
    /***************************************************************************
     * Calculate R² Score (Coefficient of Determination)
     ***************************************************************************/
    float calculateR2(float* predicted, float* actual, int count);
};

/*******************************************************************************
 * IMPLEMENTATION
 ******************************************************************************/

// Constructor
FabricTempSensor::FabricTempSensor() {
    _initialized = false;
    _prediction_count = 0;
}

// Initialization
void FabricTempSensor::begin() {
    _initialized = true;
    _prediction_count = 0;
    
    resetStats(&_resistance_stats);
    resetStats(&_env_temp_stats);
    resetStats(&_humidity_stats);
    resetStats(&_prediction_stats);
}

// Reset statistics structure
void FabricTempSensor::resetStats(SensorStats* stats) {
    stats->mean = 0.0f;
    stats->stddev = 0.0f;
    stats->min = INFINITY;
    stats->max = -INFINITY;
    stats->count = 0;
    stats->sum = 0.0f;
    stats->sum_squared = 0.0f;
}

// Update statistics with new value
void FabricTempSensor::updateStats(SensorStats* stats, float value) {
    stats->count++;
    stats->sum += value;
    stats->sum_squared += value * value;
    
    if (value < stats->min) stats->min = value;
    if (value > stats->max) stats->max = value;
    
    stats->mean = stats->sum / stats->count;
    
    if (stats->count > 1) {
        float variance = (stats->sum_squared / stats->count) - (stats->mean * stats->mean);
        stats->stddev = sqrt(fabs(variance));
    }
}

// Efficient power calculation for small integer exponents
float FabricTempSensor::calculatePower(float base, uint8_t exponent) {
    if (exponent == 0) return 1.0f;
    if (exponent == 1) return base;
    
    float result = 1.0f;
    for (uint8_t i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

// Resistance scaling with calibration
float FabricTempSensor::scaleResistance(float resistance) {
    if (resistance > RESISTANCE_SCALE_THRESHOLD) {
        return resistance - RESISTANCE_SCALE_OFFSET;
    }
    return resistance;
}

// Core prediction function
float FabricTempSensor::predictTemperature(float resistance, float env_temp, float humidity) {
    if (!_initialized) return NAN;
    
    // Apply resistance scaling
    float scaled_resistance = scaleResistance(resistance);
    
    // Store features in array
    float features[NUM_FEATURES] = {scaled_resistance, env_temp, humidity};
    
    // Start with intercept
    float result = POLY_INTERCEPT;
    
    // Compute polynomial terms
    for (int i = 0; i < NUM_TERMS; i++) {
        float term_value = POLY_TERMS[i].coefficient;
        
        for (int j = 0; j < NUM_FEATURES; j++) {
            uint8_t power = POLY_TERMS[i].powers[j];
            if (power > 0) {
                term_value *= calculatePower(features[j], power);
            }
        }
        
        result += term_value;
    }
    
    // Update statistics
    updateStats(&_resistance_stats, resistance);
    updateStats(&_env_temp_stats, env_temp);
    updateStats(&_humidity_stats, humidity);
    updateStats(&_prediction_stats, result);
    _prediction_count++;
    
    return result;
}

// Advanced prediction with detailed result
PredictionResult FabricTempSensor::predictTemperatureAdvanced(float resistance, float env_temp, float humidity) {
    PredictionResult result;
    result.computation_time_us = 0;
    result.confidence = 0.0f;
    result.error_code = ERROR_NONE;
    result.is_valid = true;
    
    // Check initialization
    if (!_initialized) {
        result.error_code = ERROR_SENSOR_NOT_INITIALIZED;
        result.is_valid = false;
        result.temperature = NAN;
        return result;
    }
    
    // Validate inputs
    if (!validateResistance(resistance)) {
        result.error_code = ERROR_INVALID_RESISTANCE;
        result.is_valid = false;
        result.temperature = NAN;
        return result;
    }
    
    if (!validateEnvTemp(env_temp)) {
        result.error_code = ERROR_INVALID_ENV_TEMP;
        result.is_valid = false;
        result.temperature = NAN;
        return result;
    }
    
    if (!validateHumidity(humidity)) {
        result.error_code = ERROR_INVALID_HUMIDITY;
        result.is_valid = false;
        result.temperature = NAN;
        return result;
    }
    
    // Measure computation time
    uint32_t start_time = micros();
    
    // Perform prediction
    result.temperature = predictTemperature(resistance, env_temp, humidity);
    
    uint32_t end_time = micros();
    result.computation_time_us = end_time - start_time;
    
    // Validate prediction
    if (!isValidPrediction(result.temperature)) {
        result.error_code = ERROR_PREDICTION_OUT_OF_BOUNDS;
        result.is_valid = false;
        return result;
    }
    
    // Calculate confidence score (simplified heuristic)
    // Confidence is higher when inputs are in typical operating range
    float r_conf = 1.0f - fabs(resistance - 8000.0f) / 12000.0f;
    float t_conf = 1.0f - fabs(env_temp - 25.0f) / 35.0f;
    float h_conf = 1.0f - fabs(humidity - 50.0f) / 50.0f;
    
    r_conf = constrain(r_conf, 0.0f, 1.0f);
    t_conf = constrain(t_conf, 0.0f, 1.0f);
    h_conf = constrain(h_conf, 0.0f, 1.0f);
    
    result.confidence = (r_conf + t_conf + h_conf) / 3.0f;
    
    return result;
}

// Validation functions
bool FabricTempSensor::validateResistance(float resistance) {
    return (resistance >= DEFAULT_RESISTANCE_MIN && 
            resistance <= DEFAULT_RESISTANCE_MAX && 
            !isnan(resistance));
}

bool FabricTempSensor::validateEnvTemp(float env_temp) {
    return (env_temp >= ENV_TEMP_MIN && 
            env_temp <= ENV_TEMP_MAX && 
            !isnan(env_temp));
}

bool FabricTempSensor::validateHumidity(float humidity) {
    return (humidity >= HUMIDITY_MIN && 
            humidity <= HUMIDITY_MAX && 
            !isnan(humidity));
}

bool FabricTempSensor::isValidPrediction(float temperature) {
    return (temperature >= PREDICTED_TEMP_MIN && 
            temperature <= PREDICTED_TEMP_MAX && 
            !isnan(temperature));
}

// Statistics getters
SensorStats FabricTempSensor::getResistanceStats() { return _resistance_stats; }
SensorStats FabricTempSensor::getEnvTempStats() { return _env_temp_stats; }
SensorStats FabricTempSensor::getHumidityStats() { return _humidity_stats; }
SensorStats FabricTempSensor::getPredictionStats() { return _prediction_stats; }
uint32_t FabricTempSensor::getPredictionCount() { return _prediction_count; }

void FabricTempSensor::resetStatistics() {
    resetStats(&_resistance_stats);
    resetStats(&_env_temp_stats);
    resetStats(&_humidity_stats);
    resetStats(&_prediction_stats);
    _prediction_count = 0;
}

// Model information getters
const char* FabricTempSensor::getModelName() { return MODEL_NAME; }
const char* FabricTempSensor::getModelVersion() { return MODEL_TRAINING_DATE; }
const char* FabricTempSensor::getLibraryVersion() { return FABRIC_TEMP_SENSOR_VERSION; }
float FabricTempSensor::getModelMAE() { return MODEL_TEST_MAE; }
float FabricTempSensor::getModelR2() { return MODEL_TEST_R2; }
int FabricTempSensor::getModelTermCount() { return NUM_TERMS; }

// Print model information
void FabricTempSensor::printModelInfo() {
    Serial.println("\n╔══════════════════════════════════════════════════════╗");
    Serial.println("║       FABRIC TEMPERATURE SENSOR - MODEL INFO         ║");
    Serial.println("╚══════════════════════════════════════════════════════╝");
    Serial.printf("Library Version:    %s\n", getLibraryVersion());
    Serial.printf("Model Name:         %s\n", getModelName());
    Serial.printf("Model Date:         %s\n", getModelVersion());
    Serial.printf("Polynomial Degree:  %d\n", POLY_DEGREE);
    Serial.printf("Number of Terms:    %d\n", NUM_TERMS);
    Serial.printf("Test MAE:           %.3f°C\n", getModelMAE());
    Serial.printf("Test R²:            %.4f\n", getModelR2());
    Serial.printf("Training Samples:   %d\n", MODEL_TRAINING_SAMPLES);
    Serial.println("══════════════════════════════════════════════════════\n");
}

// Print statistics
void FabricTempSensor::printStatistics() {
    Serial.println("\n╔══════════════════════════════════════════════════════╗");
    Serial.println("║            SENSOR STATISTICS SUMMARY                 ║");
    Serial.println("╚══════════════════════════════════════════════════════╝");
    Serial.printf("Total Predictions:  %u\n\n", _prediction_count);
    
    Serial.println("RESISTANCE (Ω):");
    Serial.printf("  Mean:    %.2f\n", _resistance_stats.mean);
    Serial.printf("  Std Dev: %.2f\n", _resistance_stats.stddev);
    Serial.printf("  Min:     %.2f\n", _resistance_stats.min);
    Serial.printf("  Max:     %.2f\n\n", _resistance_stats.max);
    
    Serial.println("ENVIRONMENTAL TEMPERATURE (°C):");
    Serial.printf("  Mean:    %.2f\n", _env_temp_stats.mean);
    Serial.printf("  Std Dev: %.2f\n", _env_temp_stats.stddev);
    Serial.printf("  Min:     %.2f\n", _env_temp_stats.min);
    Serial.printf("  Max:     %.2f\n\n", _env_temp_stats.max);
    
    Serial.println("HUMIDITY (%):");
    Serial.printf("  Mean:    %.2f\n", _humidity_stats.mean);
    Serial.printf("  Std Dev: %.2f\n", _humidity_stats.stddev);
    Serial.printf("  Min:     %.2f\n", _humidity_stats.min);
    Serial.printf("  Max:     %.2f\n\n", _humidity_stats.max);
    
    Serial.println("PREDICTIONS (°C):");
    Serial.printf("  Mean:    %.2f\n", _prediction_stats.mean);
    Serial.printf("  Std Dev: %.2f\n", _prediction_stats.stddev);
    Serial.printf("  Min:     %.2f\n", _prediction_stats.min);
    Serial.printf("  Max:     %.2f\n", _prediction_stats.max);
    Serial.println("══════════════════════════════════════════════════════\n");
}

// Print prediction result
void FabricTempSensor::printPredictionResult(PredictionResult result) {
    Serial.println("\n─────────────── PREDICTION RESULT ───────────────");
    
    if (result.is_valid) {
        Serial.printf("Temperature:     %.2f°C\n", result.temperature);
        Serial.printf("Confidence:      %.2f%%\n", result.confidence * 100);
        Serial.printf("Computation Time: %u μs\n", result.computation_time_us);
        Serial.println("Status:          ✓ VALID");
    } else {
        Serial.println("Status:          ✗ INVALID");
        Serial.print("Error Code:      ");
        switch(result.error_code) {
            case ERROR_INVALID_RESISTANCE:
                Serial.println("Invalid Resistance");
                break;
            case ERROR_INVALID_ENV_TEMP:
                Serial.println("Invalid Environmental Temperature");
                break;
            case ERROR_INVALID_HUMIDITY:
                Serial.println("Invalid Humidity");
                break;
            case ERROR_PREDICTION_OUT_OF_BOUNDS:
                Serial.println("Prediction Out of Bounds");
                break;
            case ERROR_SENSOR_NOT_INITIALIZED:
                Serial.println("Sensor Not Initialized");
                break;
            default:
                Serial.println("Unknown Error");
        }
    }
    Serial.println("─────────────────────────────────────────────────\n");
}

// Calculate Mean Absolute Error
float FabricTempSensor::calculateMAE(float* predicted, float* actual, int count) {
    if (count <= 0) return NAN;
    
    float sum_error = 0.0f;
    for (int i = 0; i < count; i++) {
        sum_error += fabs(predicted[i] - actual[i]);
    }
    return sum_error / count;
}

// Calculate Root Mean Squared Error
float FabricTempSensor::calculateRMSE(float* predicted, float* actual, int count) {
    if (count <= 0) return NAN;
    
    float sum_squared_error = 0.0f;
    for (int i = 0; i < count; i++) {
        float error = predicted[i] - actual[i];
        sum_squared_error += error * error;
    }
    return sqrt(sum_squared_error / count);
}

// Calculate R² Score
float FabricTempSensor::calculateR2(float* predicted, float* actual, int count) {
    if (count <= 0) return NAN;
    
    // Calculate mean of actual values
    float mean_actual = 0.0f;
    for (int i = 0; i < count; i++) {
        mean_actual += actual[i];
    }
    mean_actual /= count;
    
    // Calculate total sum of squares and residual sum of squares
    float ss_total = 0.0f;
    float ss_residual = 0.0f;
    
    for (int i = 0; i < count; i++) {
        float error = actual[i] - predicted[i];
        float deviation = actual[i] - mean_actual;
        
        ss_residual += error * error;
        ss_total += deviation * deviation;
    }
    
    // R² = 1 - (SS_residual / SS_total)
    if (ss_total == 0.0f) return NAN;
    return 1.0f - (ss_residual / ss_total);
}

/*******************************************************************************
 * BACKWARD COMPATIBILITY FUNCTIONS
 * 
 * These functions maintain compatibility with the original TempSensorCalibration.h
 * interface, allowing existing code to work without modifications.
 ******************************************************************************/

#undef mean
#undef stddev
#undef meanT
#undef stddevT

float RScale(int x, float mean, float stddev)
{
    return (x - mean) / stddev;
}


/**
 * Legacy temperature estimation function (C-style)
 * 
 * This is the original function signature from TempSensorCalibration.h
 * It performs the same polynomial calculation but as a standalone function.
 * 
 * Parameters:
 *   R     - Fabric sensor resistance (Ω)
 *   T_env - Environmental temperature (°C)
 *   H     - Relative humidity (%)
 * 
 * Returns:
 *   Predicted fabric temperature (°C)
 */
inline float estimate_temperature(float R, float T_env, float H) {
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

/**
 * Legacy feature definitions for backward compatibility
 */
#define FEATURE_R FEATURE_RESISTANCE
#define FEATURE_T_ENV FEATURE_ENV_TEMP
#define FEATURE_H FEATURE_HUMIDITY

/*******************************************************************************
 * ADDITIONAL UTILITY FUNCTIONS
 ******************************************************************************/

/**
 * Convert ADC reading to resistance
 * 
 * Calculates sensor resistance from voltage divider circuit
 * 
 * Parameters:
 *   adc_value - ADC reading (0-4095 for 12-bit ADC)
 *   vcc       - Supply voltage (typically 3.3V)
 *   adc_max   - Maximum ADC value (typically 4095)
 *   r_ref     - Reference resistor value (Ω)
 * 
 * Returns:
 *   Calculated resistance (Ω)
 */
inline float adcToResistance(int adc_value, float vcc, int adc_max, float r_ref) {
    if (adc_value <= 0 || adc_value >= adc_max) {
        return NAN;
    }
    
    float vout = (adc_value * vcc) / adc_max;
    return r_ref * ((vcc / vout) - 1.0f);
}

/**
 * Convert resistance to ADC value (inverse calculation)
 * 
 * Useful for simulation and testing
 * 
 * Parameters:
 *   resistance - Sensor resistance (Ω)
 *   vcc        - Supply voltage (V)
 *   adc_max    - Maximum ADC value
 *   r_ref      - Reference resistor value (Ω)
 * 
 * Returns:
 *   Expected ADC reading
 */
inline int resistanceToAdc(float resistance, float vcc, int adc_max, float r_ref) {
    if (resistance <= 0) return 0;
    
    float vout = (vcc * r_ref) / (resistance + r_ref);
    return (int)((vout * adc_max) / vcc);
}

/**
 * Temperature error calculation
 * 
 * Computes absolute error between predicted and actual temperature
 * 
 * Parameters:
 *   predicted - Predicted temperature (°C)
 *   actual    - Actual/measured temperature (°C)
 * 
 * Returns:
 *   Absolute error (°C)
 */
inline float calculateTemperatureError(float predicted, float actual) {
    return fabs(predicted - actual);
}

/**
 * Percent error calculation
 * 
 * Parameters:
 *   predicted - Predicted temperature (°C)
 *   actual    - Actual/measured temperature (°C)
 * 
 * Returns:
 *   Percent error (%)
 */
inline float calculatePercentError(float predicted, float actual) {
    if (actual == 0.0f) return NAN;
    return (fabs(predicted - actual) / fabs(actual)) * 100.0f;
}

/**
 * Check if prediction is within acceptable error margin
 * 
 * Parameters:
 *   predicted      - Predicted temperature (°C)
 *   actual         - Actual temperature (°C)
 *   error_margin   - Acceptable error margin (°C)
 * 
 * Returns:
 *   true if within margin, false otherwise
 */
inline bool isWithinErrorMargin(float predicted, float actual, float error_margin) {
    return (calculateTemperatureError(predicted, actual) <= error_margin);
}

/**
 * Simple moving average filter for temperature smoothing
 * 
 * Parameters:
 *   new_value  - New temperature reading (°C)
 *   buffer     - Array of previous readings
 *   buffer_size - Size of the buffer
 *   index      - Current buffer index (will be updated)
 * 
 * Returns:
 *   Filtered temperature (°C)
 */
inline float movingAverageFilter(float new_value, float* buffer, int buffer_size, int* index) {
    buffer[*index] = new_value;
    *index = (*index + 1) % buffer_size;
    
    float sum = 0.0f;
    for (int i = 0; i < buffer_size; i++) {
        sum += buffer[i];
    }
    
    return sum / buffer_size;
}

/**
 * Exponential moving average filter
 * 
 * Provides faster response to changes compared to simple moving average
 * 
 * Parameters:
 *   new_value    - New temperature reading (°C)
 *   prev_average - Previous average value (°C)
 *   alpha        - Smoothing factor (0-1, higher = less smoothing)
 * 
 * Returns:
 *   Filtered temperature (°C)
 */
inline float exponentialMovingAverage(float new_value, float prev_average, float alpha) {
    return (alpha * new_value) + ((1.0f - alpha) * prev_average);
}

/*******************************************************************************
 * CALIBRATION AND TESTING UTILITIES
 ******************************************************************************/

/**
 * Generate test prediction
 * 
 * Useful for validating model performance
 * 
 * Parameters:
 *   resistance - Test resistance value (Ω)
 *   env_temp   - Test environmental temperature (°C)
 *   humidity   - Test humidity (%)
 *   actual_temp - Known actual temperature (°C)
 * 
 * Prints comparison to Serial
 */
inline void testPrediction(float resistance, float env_temp, float humidity, float actual_temp) {
    float predicted = estimate_temperature(resistance, env_temp, humidity);
    float error = calculateTemperatureError(predicted, actual_temp);
    float percent_error = calculatePercentError(predicted, actual_temp);
    
    Serial.println("\n─── Test Prediction ───");
    Serial.printf("Inputs:    R=%.1fΩ, T_env=%.1f°C, H=%.1f%%\n", resistance, env_temp, humidity);
    Serial.printf("Predicted: %.2f°C\n", predicted);
    Serial.printf("Actual:    %.2f°C\n", actual_temp);
    Serial.printf("Error:     %.2f°C (%.1f%%)\n", error, percent_error);
    Serial.println("───────────────────────\n");
}

/**
 * Batch prediction testing
 * 
 * Tests multiple data points and calculates overall statistics
 * 
 * Parameters:
 *   resistance_array - Array of resistance values (Ω)
 *   env_temp_array   - Array of environmental temperatures (°C)
 *   humidity_array   - Array of humidity values (%)
 *   actual_temp_array - Array of actual temperatures (°C)
 *   count            - Number of test samples
 * 
 * Prints performance metrics to Serial
 */
inline void batchTestPredictions(float* resistance_array, float* env_temp_array, 
                                  float* humidity_array, float* actual_temp_array, int count) {
    if (count <= 0) return;
    
    float* predictions = new float[count];
    
    // Generate predictions
    for (int i = 0; i < count; i++) {
        predictions[i] = estimate_temperature(resistance_array[i], env_temp_array[i], humidity_array[i]);
    }
    
    // Calculate statistics
    float sum_error = 0.0f;
    float sum_squared_error = 0.0f;
    float max_error = 0.0f;
    
    for (int i = 0; i < count; i++) {
        float error = fabs(predictions[i] - actual_temp_array[i]);
        sum_error += error;
        sum_squared_error += error * error;
        
        if (error > max_error) {
            max_error = error;
        }
    }
    
    float mae = sum_error / count;
    float rmse = sqrt(sum_squared_error / count);
    
    // Calculate R²
    float mean_actual = 0.0f;
    for (int i = 0; i < count; i++) {
        mean_actual += actual_temp_array[i];
    }
    mean_actual /= count;
    
    float ss_total = 0.0f;
    float ss_residual = 0.0f;
    
    for (int i = 0; i < count; i++) {
        float error = actual_temp_array[i] - predictions[i];
        float deviation = actual_temp_array[i] - mean_actual;
        ss_residual += error * error;
        ss_total += deviation * deviation;
    }
    
    float r2 = (ss_total > 0) ? (1.0f - (ss_residual / ss_total)) : NAN;
    
    // Print results
    Serial.println("\n╔════════════════════════════════════════╗");
    Serial.println("║     BATCH TEST RESULTS                 ║");
    Serial.println("╚════════════════════════════════════════╝");
    Serial.printf("Test Samples:  %d\n", count);
    Serial.printf("MAE:           %.3f°C\n", mae);
    Serial.printf("RMSE:          %.3f°C\n", rmse);
    Serial.printf("Max Error:     %.3f°C\n", max_error);
    Serial.printf("R² Score:      %.4f\n", r2);
    Serial.println("════════════════════════════════════════\n");
    
    delete[] predictions;
}

/**
 * Print polynomial model structure
 * 
 * Displays all terms in human-readable format
 */
inline void printPolynomialModel() {
    Serial.println("\n╔══════════════════════════════════════════════════════╗");
    Serial.println("║          POLYNOMIAL MODEL STRUCTURE                  ║");
    Serial.println("╚══════════════════════════════════════════════════════╝");
    Serial.printf("T = %.4e\n", POLY_INTERCEPT);
    
    for (int i = 0; i < NUM_TERMS; i++) {
        Serial.print("    ");
        if (POLY_TERMS[i].coefficient >= 0) Serial.print("+");
        Serial.printf("%.4e", POLY_TERMS[i].coefficient);
        
        // Print term structure
        bool first = true;
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (POLY_TERMS[i].powers[j] > 0) {
                if (!first) Serial.print("×");
                first = false;
                
                switch(j) {
                    case FEATURE_RESISTANCE: Serial.print("R"); break;
                    case FEATURE_ENV_TEMP: Serial.print("T"); break;
                    case FEATURE_HUMIDITY: Serial.print("H"); break;
                }
                
                if (POLY_TERMS[i].powers[j] > 1) {
                    Serial.printf("^%d", POLY_TERMS[i].powers[j]);
                }
            }
        }
        Serial.println();
    }
    Serial.println("══════════════════════════════════════════════════════\n");
}

/*******************************************************************************
 * GLOBAL CONVENIENCE INSTANCE (OPTIONAL)
 * 
 * Uncomment the line below to create a global sensor instance that can be
 * used without declaring your own FabricTempSensor object.
 * 
 * Usage: FabricSensor.begin(); FabricSensor.predictTemperature(...);
 ******************************************************************************/
// extern FabricTempSensor FabricSensor;

#endif // FABRIC_TEMP_SENSOR_H

/*******************************************************************************
 * END OF FILE
 * 
 * This library provides a complete solution for textile-based temperature
 * sensing with embedded machine learning capabilities. It maintains full
 * backward compatibility with the original TempSensorCalibration.h while
 * offering advanced features for professional IoT applications.
 * 
 * For support and documentation, please refer to the project repository.
 ******************************************************************************/