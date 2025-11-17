/**************  Blynk Configuration  *****************/
#define BLYNK_TEMPLATE_ID "TMPL3sns_muJv"
#define BLYNK_TEMPLATE_NAME "Fabric Temperature Sensor"
#define BLYNK_AUTH_TOKEN  "S4FrXDOvbPR8o0fCIP6b5sgmzygwMP9J"

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

/**************  Sensor Libraries  ********************/
#include <Wire.h>
#include <Adafruit_HTU21DF.h>
#include <max6675.h>
#include "TempSensorCalibration.h"

/**************  WiFi Credentials  ********************/
char ssid[] = "Hello";
char pass[] = "123456789";

/**************  MAX6675 Pins  ************************/
const int thermoDO = 19;
const int thermoCS = 5;
const int thermoCLK = 18;

/**************  Fabric Sensor Config  ****************/
const int adcPin = 34;
const float Vcc = 3.3;
const int adcMax = 4095;
const float Rref = 10000.0;

/**************  Sensors Setup  ***********************/
Adafruit_HTU21DF htu = Adafruit_HTU21DF();
MAX6675 thermocouple(thermoCLK, thermoCS, thermoDO);

/**************  Timing  ******************************/
unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000;

/**************  ALERT CONTROL  ***********************/
bool alertSent = false;   // prevents spamming notifications

/**************  Setup  *******************************/
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32 - HTU21D + Fabric Sensor + MAX6675 + ML + BLYNK ===");

  Serial.println("Connecting to Blynk...");
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  Wire.begin();

  Serial.println("Initializing HTU21D...");
  if (!htu.begin())
    Serial.println("  ✗ HTU21D not found!");
  else
    Serial.println("  ✓ HTU21D found!");

  Serial.println("Initializing MAX6675...");
  delay(500);

  analogReadResolution(12);

  Serial.println("Setup complete!\n");
}

/**************  Loop  *******************************/
void loop() {
  Blynk.run();
  unsigned long currentTime = millis();

  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;

    Serial.println("========== Sensor Readings ==========");

    /************** Fabric Sensor **************/
    int adcVal = analogRead(adcPin);
    float Vout = (adcVal * Vcc) / adcMax;
    float Rsens = 0;

    if (adcVal > 0 && adcVal < adcMax) {
      Rsens = Rref * ((Vcc / Vout) - 1.0);
      Rsens = Rscale(Rsens); 
    }

    Serial.printf("Fabric Sensor - ADC: %d, Vout: %.3f V, Resistance: %.2f Ω\n",
                  adcVal, Vout, Rsens);

    /************** HTU21D **************/
    float T_env = htu.readTemperature();
    float H_env = htu.readHumidity();

    if (isnan(T_env) || isnan(H_env)) {
      Serial.println("HTU21D: Read error!");
      T_env = 25;
      H_env = 50;
    }

    Serial.printf("Environment - Temp: %.2f °C, Humidity: %.2f %%\n",
                  T_env, H_env);

    /************** MAX6675 **************/
    float expectedTemp = thermocouple.readCelsius();

    if (isnan(expectedTemp)) {
      Serial.println("MAX6675: Read error!");
      expectedTemp = -999;
    }

    Serial.printf("Expected Temperature (MAX6675): %.2f °C\n", expectedTemp);

    /************** ML MODEL PREDICTION **************/
    float fabricTemp = estimate_temperature(Rsens, 23.36, 37.75);

    Serial.printf("→ Predicted Fabric Temperature (ML): %.2f °C\n",
                  fabricTemp);

    Serial.println("=====================================\n");

    /************** SEND DATA TO BLYNK **************/
    Blynk.virtualWrite(V0, Rsens);
    Blynk.virtualWrite(V1, T_env);
    Blynk.virtualWrite(V2, H_env);
    Blynk.virtualWrite(V3, expectedTemp);
    Blynk.virtualWrite(V4, fabricTemp);

  }

  delay(100);
}
