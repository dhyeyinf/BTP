# Smart IoT-Integrated Textile-Based Temperature Sensor with AI-Driven Predictive Analytics

<div align="center">
          
![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Platform](https://img.shields.io/badge/Platform-ESP32-blue)
![ML Framework](https://img.shields.io/badge/ML-Gradient%20Boosting-orange)
![IoT](https://img.shields.io/badge/IoT-Blynk-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

*A B.Tech Project by IIT Jodhpur*

[Report](./BTP.pdf) • [Demo](#demo) • [Documentation](#documentation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Components](#hardware-components)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Repository Structure](#repository-structure)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Overview

This project demonstrates a complete end-to-end implementation of a **wearable temperature sensing system** using conductive textile materials, enhanced with **machine learning calibration** and **IoT connectivity**. The system addresses the limitations of traditional thermometers by providing:

- **Real-time continuous monitoring**
- **Flexible, wearable form factor**
- **Environmental compensation** (humidity & ambient temperature)
- **Cloud-connected IoT dashboard**
- **Ultra-low latency** (~45μs inference time)

### Academic Context

**Institution:** Indian Institute of Technology Jodhpur  
**Department:** Electrical Engineering  
**Degree:** Bachelor of Technology (B.Tech.)  
**Supervisor:** Dr. Ajay Agarwal  
**Teaching Assistant:** Ms. Anupam

---

## Key Features

### **Textile-Based Sensing**
- Conductive yarn as temperature-sensitive element
- Resistance-based measurement (2.9kΩ - 9.7kΩ range)
- Operates across 23°C - 102.5°C

### **AI-Driven Calibration**
- **Gradient Boosting Regressor** for non-linear compensation
- Multi-feature input: Resistance, Environmental Temp, Humidity
- **Test MAE: 3.09°C** | **R² Score: 0.956**
- Embedded polynomial approximation (3.7 kB model size)

### **IoT Integration**
- ESP32-based wireless connectivity
- Real-time Blynk dashboard (Web + Mobile)
- Firebase & Google Sheets data logging
- Bluetooth-enabled monitoring

### **Optimized Embedded Deployment**
- Custom C++ inference library (`FabricTempSensor`)
- 45μs prediction latency
- 196 bytes RAM usage
- 34-term polynomial regression

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SENSING LAYER                            │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐      │
│  │ Fabric      │  │ HTU21D   │  │ MAX6675           │      │
│  │ Sensor      │  │ (Env)    │  │ (Reference)       │      │
│  │ (Textile)   │  │ I²C      │  │ SPI               │      │
│  └──────┬──────┘  └────┬─────┘  └─────┬─────────────┘      │
└─────────┼──────────────┼──────────────┼────────────────────┘
          │              │              │
┌─────────▼──────────────▼──────────────▼────────────────────┐
│              ESP32 MICROCONTROLLER                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     FabricTempSensor Library (3.7 kB)                │  │
│  │  • Polynomial ML Model (34 terms)                    │  │
│  │  • Environmental Compensation                        │  │
│  │  • Input Validation & Filtering                      │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬──────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼─────┐  ┌──────▼──────┐  ┌────▼──────┐
│   Blynk     │  │  Firebase   │  │  Google   │
│  Dashboard  │  │   RTDB      │  │  Sheets   │
│ (Web+Mobile)│  │             │  │           │
└─────────────┘  └─────────────┘  └───────────┘
```

---

## Hardware Components

| Component | Model/Type | Interface | Purpose |
|-----------|-----------|-----------|---------|
| **Microcontroller** | ESP32 DevKit | - | Main processing unit |
| **Fabric Sensor** | Conductive Yarn | ADC (GPIO 34) | Temperature sensing element |
| **Environmental Sensor** | HTU21D | I²C (GPIO 21, 22) | Ambient temp & humidity |
| **Reference Sensor** | MAX6675 K-type | SPI (GPIO 18, 5, 19) | Ground truth calibration |
| **Power Supply** | USB 5V | - | System power |

### Circuit Configuration

**Voltage Divider for Resistance Measurement:**
```
R_sensor = R_ref × (V_in / V_out - 1)
where R_ref = 10kΩ
```

---

## Machine Learning Pipeline

### Data Collection
- **548 calibration samples** across 23°C - 102.5°C
- Controlled heating using laboratory hotplate
- Multi-sensor dataset (Sensor-1, Sensor-2, Sensor-3)
- Automated Google Sheets logging via ESP32

### Feature Engineering

**Steinhart-Hart Inspired Features:**
- R⁻¹, ln(R), [ln(R)]³

**Polynomial Expansion:**
- √R, R², [ln(R)]²

**Cross-Interaction Terms:**
- R·T_env, ln(R)·T_env, ln(R)·H

### Model Comparison

| Model | Feature Set | Test MAE | Test RMSE | Test R² |
|-------|-------------|----------|-----------|---------|
| **Gradient Boosting** | **Steinhart-Hart** | **3.09°C** | **5.22°C** | **0.956** |
| Random Forest | Simple | 3.74°C | 5.46°C | 0.952 |
| Linear Regression | Comprehensive | 4.89°C | 6.98°C | 0.926 |

### Embedded Optimization

**Polynomial Approximation Formula:**
```
T_predicted = β₀ + Σ(i+j+k≤4) β_ijk · Rⁱ · T_env^j · H^k
```
- **34 coefficients**
- **Model size: 2.1 kB** (compressed to 3.7 kB library)
- **Inference: 45μs** on ESP32

---

## 📦 Installation & Setup

### Prerequisites

```bash
# Arduino IDE 2.3.2 or later
# ESP32 Board Support (via Board Manager)
# Required Libraries (via Library Manager):
- Adafruit HTU21DF Library
- MAX6675 library
- Blynk library
- Firebase ESP32 Client
```

### Hardware Setup

1. **Connect Fabric Sensor:**
   - One end to GPIO 34 (ADC pin)
   - Other end through 10kΩ resistor to GND
   - Junction to 3.3V

2. **Connect HTU21D (I²C):**
   - SDA → GPIO 21
   - SCL → GPIO 22
   - VCC → 3.3V, GND → GND

3. **Connect MAX6675 (SPI):**
   - SCK → GPIO 18
   - CS → GPIO 5
   - DO → GPIO 19
   - VCC → 3.3V, GND → GND

### Software Installation

```bash
# 1. Clone the repository
git clone https://github.com/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics.git

# 2. Open Arduino IDE and install ESP32 board support
# File → Preferences → Additional Board Manager URLs:
# https://dl.espressif.com/dl/package_esp32_index.json

# 3. Install required libraries through Library Manager

# 4. Configure WiFi and Blynk credentials in the code
# Edit: WIFI_SSID, WIFI_PASSWORD, BLYNK_AUTH_TOKEN

# 5. Upload to ESP32
```

---

## 💻 Usage

### Basic Example

```cpp
#include "FabricTempSensor.h"

FabricTempSensor sensor;

void setup() {
  Serial.begin(115200);
  sensor.begin();
}

void loop() {
  // Read sensor values
  float resistance = sensor.readResistance();
  float envTemp = sensor.readEnvTemperature();
  float humidity = sensor.readHumidity();
  
  // Get calibrated temperature
  float temperature = sensor.predictTemperature(resistance, envTemp, humidity);
  
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  delay(1000);
}
```

### Advanced Usage with Validation

```cpp
PredictionResult result = sensor.predictTemperatureAdvanced(R, T_env, H);

if (result.is_valid) {
  Serial.printf("Temperature: %.2f°C\n", result.temperature);
  Serial.printf("Confidence: %.2f%%\n", result.confidence * 100);
  Serial.printf("Computation Time: %d μs\n", result.computation_time_us);
} else {
  Serial.printf("Error Code: %d\n", result.error_code);
}
```

---

## 📊 Results & Performance

### Overall System Performance

| Metric | Value |
|--------|-------|
| **Temperature Range** | 23°C - 102.5°C |
| **Overall MAE** | 0.83°C (full dataset) |
| **Inference Latency** | 45μs |
| **Model Size** | 3.7 kB |
| **RAM Usage** | 196 bytes |
| **WiFi Range** | 50+ meters |

### Temperature Zone Accuracy

| Range | MAE | RMSE | Application |
|-------|-----|------|-------------|
| **30-40°C** | **0.11°C** | **0.15°C** | Wearable health monitoring |
| 40-60°C | 1.59°C | 2.18°C | Fever detection |
| 60-80°C | 3.15°C | 4.82°C | Industrial monitoring |
| 80-110°C | 2.08°C | 3.45°C | Thermal management |

### Multi-Sensor Consistency

| Sensor ID | Samples | MAE | RMSE | R² |
|-----------|---------|-----|------|----|
| Sensor-1 | 100 | 2.47°C | 5.15°C | 0.961 |
| Sensor-2 | 100 | 1.82°C | 4.28°C | 0.972 |
| **Sensor-3** | **110** | **0.83°C** | **2.44°C** | **0.989** |

### Environmental Robustness

| Condition | Temp | Humidity | MAE | RMSE |
|-----------|------|----------|-----|------|
| Dry, Cool | 20°C | 30% | 0.92°C | 2.58°C |
| Normal | 23°C | 50% | 0.83°C | 2.44°C |
| Humid, Warm | 27°C | 70% | 0.68°C | 2.12°C |

---

## 📁 Repository Structure

```
Smart-IoT-Integrated-Textile-Based-Temperature-Sensor/
├── README.md                          # This file
├── BTP.pdf                            # Complete project report
│
├── data/                              # Calibration datasets
│   ├── sensor1_data.csv
│   ├── sensor2_data.csv
│   └── sensor3_data.csv
│
├── notebooks/                         # Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   ├── ml_calibration.ipynb
│   └── model_evaluation.ipynb
│
├── src/                               # Source code
│   ├── arduino/
│   │   ├── main.ino                  # Main ESP32 firmware
│   │   ├── FabricTempSensor.h        # ML inference library (header)
│   │   └── FabricTempSensor.cpp      # ML inference library (implementation)
│   │
│   ├── scripts/
│   │   ├── firebase_logger.py        # Firebase data logging
│   │   └── google_sheets_logger.gs   # Google Apps Script
│   │
│   └── models/
│       ├── gradient_boosting.pkl     # Trained scikit-learn model
│       └── polynomial_coefficients.h  # Embedded model coefficients
│
├── hardware/                          # Hardware design files
│   ├── circuit_diagram.png
│   └── sensor_placement.jpg
│
├── docs/                              # Additional documentation
│   ├── SETUP_GUIDE.md
│   ├── API_REFERENCE.md
│   └── TROUBLESHOOTING.md
│
└── results/                           # Performance analysis
    ├── performance_plots/
    └── test_results.json
```

---

## 👥 Team

<table>
  <tr>
    <td align="center">
      <strong>Dhyey Findoriya</strong><br>
      <em>B22EE024</em><br>
      Machine Learning Calibration<br>
      <a href="https://github.com/dhyeyinf">GitHub</a>
    </td>
    <td align="center">
      <strong>Lavangi Parihar</strong><br>
      <em>B22EE044</em><br>
      Embedded Library Development<br>
      <a href="https://github.com/Lavangip">GitHub</a>
    </td>
    <td align="center">
      <strong>Vaghela Nisarg Nirav</strong><br>
      <em>B22EE068</em><br>
      Experimental Setup & Data Collection<br>
      <a href="https://github.com/nisarg-vaghela17">GitHub</a>
    </td>
  </tr>
</table>

### Individual Contributions

**Chapter 2 - Experimental Setup** (Vaghela Nisarg Nirav)
- Hardware interfacing and circuit design
- Automated data logging pipeline
- Multi-sensor calibration dataset generation

**Chapter 3 - Machine Learning Calibration** (Dhyey Findoriya)
- Feature engineering and model selection
- Comparative analysis of 20 model-feature combinations
- Polynomial model optimization for embedded deployment

**Chapter 4 - Embedded Library Development** (Lavangi Parihar)
- C++ inference library implementation
- Real-time performance validation
- IoT dashboard integration

---

## 🙏 Acknowledgments

We express our sincere gratitude to:

- **Dr. Ajay Agarwal** - Project Supervisor, for invaluable guidance and support
- **Ms. Anupam** - Teaching Assistant, for consistent guidance throughout the project
- **Department of Electrical Engineering, IIT Jodhpur** - For providing facilities and resources
- **Open-source community** - For the libraries and tools that made this project possible

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@techreport{findoriya2025textile,
  title={Smart IoT-Integrated Textile-Based Temperature Sensor with AI-Driven Predictive Analytics},
  author={Findoriya, Dhyey and Parihar, Lavangi and Vaghela, Nisarg Nirav},
  institution={Indian Institute of Technology Jodhpur},
  year={2025},
  type={B.Tech. Project Report},
  department={Department of Electrical Engineering}
}
```

---

## 🔮 Future Work

### Advanced Calibration
- Extended environmental range testing (20-90% humidity)
- LSTM/GRU networks for temporal modeling
- Transfer learning for cross-sensor generalization

### Hardware Enhancement
- Integration with flexible printed electronics
- Ultra-low-power optimization for wearable applications
- Multi-node mesh networking (LoRa/ESP-NOW)

### Application Development
- Medical-grade wearable health monitoring
- Smart textile integration for adaptive clothing
- Industrial thermal management systems

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions, collaborations, or feedback:

- **GitHub Issues:** [Create an issue](https://github.com/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics/issues)
- **Email:** Contact through IIT Jodhpur official channels
- **Project Report:** [BTP.pdf](./BTP.pdf)

---

<div align="center">

**Made with ❤️ at IIT Jodhpur**

*Department of Electrical Engineering • Indian Institute of Technology Jodhpur • 2025*

[![GitHub stars](https://img.shields.io/github/stars/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics?style=social)](https://github.com/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics?style=social)](https://github.com/dhyeyinf/Smart-IoT-Integrated-Textile-Based-Temperature-Sensor-with-AI-Driven-Predictive-Analytics/network/members)

</div>
