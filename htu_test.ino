#include <Wire.h>
#include "Adafruit_HTU21DF.h"

Adafruit_HTU21DF htu = Adafruit_HTU21DF();

// Custom sensor setup with voltage divider
const int adcPin = 4;         // Using GPIO4 (ADC2 channel 0)
const float Vcc = 3.3;        // ESP32 supply voltage
const int adcMax = 4095;      // 12-bit ADC
const float Rref = 10000.0;   // 10k reference resistor

void setup() {
  Serial.begin(115200);

  // HTU21D init
  if (!htu.begin()) {
    Serial.println("Check circuit. HTU21D not found!");
    while (1);
  }

  // ADC setup
  analogReadResolution(12);   // 12-bit ADC (0–4095)
}

void loop() {
  // --- HTU21D readings ---
  float temp = htu.readTemperature();
  float hum = htu.readHumidity();
  Serial.print("HTU21D -> Temperature(°C): ");
  Serial.print(temp);
  Serial.print("\t Humidity(%): ");
  Serial.println(hum);

  // --- Custom sensor readings ---
  int adcVal = analogRead(adcPin);
  float Vout = (adcVal * Vcc) / adcMax;

  if (adcVal == 0) {
    Serial.println("Custom Sensor -> Open circuit detected!");
  } else if (adcVal == adcMax) {
    Serial.println("Custom Sensor -> Short circuit detected!");
  } else {
    // Calculate sensor resistance from divider
    float Rsens = (Rref * (Vcc - Vout)) / Vout;

    Serial.print("Custom Sensor -> ADC Value: ");
    Serial.print(adcVal);
    Serial.print("  -> Voltage: ");
    Serial.print(Vout, 3);
    Serial.print(" V  -> Sensor Resistance: ");
    Serial.print(Rsens, 2);
    Serial.println(" ohms");
  }

  Serial.println("-------------------------------------------------");
  delay(2000);
}
