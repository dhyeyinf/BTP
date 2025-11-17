#define BLYNK_TEMPLATE_ID "TMPL3sns_muJv"
#define BLYNK_TEMPLATE_NAME "Fabric Temperature Sensor"


#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_HTU21DF.h>

////////// === BLYNK ADDED ===
#include <BlynkSimpleEsp32.h>
// --- Add these BEFORE including any Blynk headers ---


// (optional) device auth token macro is not required but you already have BLYNK_AUTH char[] below

BlynkTimer timer;
char BLYNK_AUTH[] = "S4FrXDOvbPR8o0fCIP6b5sgmzygwMP9J"; // <-- put your token here
// Virtual pin mapping:
// V0 -> custom sensor resistance (Ohm)
// V1 -> HTU21D temperature (°C)
// V2 -> HTU21D humidity (%)
// V3 -> MAX6675 temperature (°C)
////////// === END BLYNK ADDED ===

// --- WiFi Credentials ---
const char* ssid = "Hello";
const char* password = "123456789";

// --- Google Apps Script Web App URL ---
const char* serverName = "https://script.google.com/macros/s/AKfycby-06__uSoujD_AFGIrv3ua4XTzz7Zb8b85tKF-PAJ-a2tUQll-i0uM_NChhNxvnXbO/exec";

// --- MAX6675 pins ---
const int thermoDO = 19;   // SO (MISO)
const int thermoCS = 5;    // CS
const int thermoCLK = 18;  // SCK

// --- Custom Sensor setup ---
const int adcPin = 34;
const float Vcc = 3.3;
const int adcMax = 4095;
const float Rref = 10000.0;

// --- HTU21D sensor ---
Adafruit_HTU21DF htu = Adafruit_HTU21DF();

// --- Timing ---
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 10000;
unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000;

// === Global sensor variables (so Blynk timer callback can access latest values) ===
float g_thermoTemp = NAN;
int   g_adcVal = 0;
float g_Vout = NAN;
float g_Rsens = NAN;
float g_envTemp = NAN;
float g_envHumidity = NAN;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32 Multi-Sensor Logger ===");

  // Setup MAX6675 pins
  pinMode(thermoCS, OUTPUT);
  pinMode(thermoCLK, OUTPUT);
  pinMode(thermoDO, INPUT);

  digitalWrite(thermoCS, HIGH);  // CS high to start conversion
  digitalWrite(thermoCLK, LOW);

  // Setup I2C for HTU21D (default pins: SDA=21, SCL=22)
  Wire.begin();

  // Initialize HTU21D
  Serial.println("Initializing HTU21D...");
  if (!htu.begin()) {
    Serial.println("  ✗ HTU21D not found! Check wiring.");
    Serial.println("  → SDA should be on GPIO 21");
    Serial.println("  → SCL should be on GPIO 22");
    Serial.println("  → VCC should be 3.3V");
  } else {
    Serial.println("  ✓ HTU21D found!");
  }

  // Setup WiFi
  connectWiFi();

  // Setup custom sensor
  analogReadResolution(12);

  // Initialize MAX6675
  Serial.println("Initializing MAX6675...");
  delay(300); // Wait for first conversion
  readMAX6675(); // Dummy read
  delay(300); // Wait for next conversion

  // === Initialize Blynk (non-invasive) ===
  // If WiFi is connected, configure and attempt Blynk connection.
  Blynk.config(BLYNK_AUTH); // config only (won't reset your WiFi)
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Connecting to Blynk...");
    // Try a brief connect; Blynk.run() in loop will maintain connection.
    if (Blynk.connect(3000)) {
      Serial.println("Blynk connected.");
    } else {
      Serial.println("Blynk: initial connect timed out, will retry in background.");
    }
  } else {
    Serial.println("WiFi not connected — Blynk will attempt when WiFi is up.");
  }

  // Setup Blynk timer to push sensor values every 2s
  timer.setInterval(2000L, pushToBlynk);

  Serial.println("Setup complete!\n");
}

void loop() {
  // Keep Blynk running (non-blocking)
  Blynk.run();
  timer.run();

  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Reconnecting...");
    connectWiFi();
    // After re-connecting WiFi, try to reconnect Blynk
    if (WiFi.status() == WL_CONNECTED) {
      Blynk.connect(3000);
    }
  }

  unsigned long currentTime = millis();

  // Read sensors at defined interval
  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;

    // --- Read MAX6675 ---
    float thermoTemp = readMAX6675();
    g_thermoTemp = thermoTemp; // store global copy

    // --- Read Custom Sensor ---
    int adcVal = analogRead(adcPin);
    g_adcVal = adcVal;
    float Vout = (adcVal * Vcc) / adcMax;
    g_Vout = Vout;
    float Rsens = 0;

    if (adcVal > 0 && adcVal < adcMax) {
      Rsens = Rref * ((Vcc / Vout) - 1.0);
      g_Rsens = Rsens;
    } else {
      g_Rsens = NAN;
    }

    // --- Read HTU21D (Environment) ---
    float envTemp = htu.readTemperature();
    float envHumidity = htu.readHumidity();
    g_envTemp = envTemp;
    g_envHumidity = envHumidity;

    // --- Display readings ---
    Serial.println("========== Sensor Readings ==========");

    // MAX6675
    if (isnan(thermoTemp)) {
      Serial.println("MAX6675: No thermocouple detected!");
      // keep g_thermoTemp as NAN
    } else {
      Serial.print("MAX6675 Temp: ");
      Serial.print(thermoTemp);
      Serial.println(" °C");
    }

    // Custom Sensor
    Serial.print("Custom Sensor - ADC: ");
    Serial.print(adcVal);
    Serial.print(", Vout: ");
    Serial.print(Vout, 3);
    Serial.print(" V, Resistance: ");
    Serial.print(Rsens, 2);
    Serial.println(" Ω");

    // HTU21D Environment
    if (isnan(envTemp) || isnan(envHumidity)) {
      Serial.println("HTU21D: Read error!");
      // keep g_envTemp / g_envHumidity as read (NaN if failed)
    } else {
      Serial.print("Environment - Temp: ");
      Serial.print(envTemp);
      Serial.print(" °C, Humidity: ");
      Serial.print(envHumidity);
      Serial.println(" %");
    }

    Serial.println("=====================================\n");

    // --- Send to Google Sheets ---
    if (currentTime - lastSendTime >= sendInterval) {
      sendToGoogleSheets(thermoTemp, adcVal, Vout, Rsens, envTemp, envHumidity);
      lastSendTime = currentTime;
    }
  }

  delay(100);
}

float readMAX6675() {
  uint16_t data = 0;

  // Pull CS low to start reading (stops conversion)
  digitalWrite(thermoCS, LOW);
  delayMicroseconds(100); // Wait for CS setup time

  // Read 16 bits (MSB first)
  for (int i = 0; i < 16; i++) {
    digitalWrite(thermoCLK, HIGH);
    delayMicroseconds(1);

    data <<= 1;
    if (digitalRead(thermoDO)) {
      data |= 1;
    }

    digitalWrite(thermoCLK, LOW);
    delayMicroseconds(1);
  }

  // Pull CS high to start new conversion
  digitalWrite(thermoCS, HIGH);

  // Debug: print raw data
  Serial.print("  [DEBUG] MAX6675 Raw: 0x");
  Serial.print(data, HEX);

  // Check bit D2 (open thermocouple detection)
  if (data & 0x0004) {
    Serial.println(" - Thermocouple OPEN");
    return NAN;
  }

  // Extract temperature (bits D14-D3, shift right by 3)
  uint16_t tempData = data >> 3;

  Serial.print(" - Temp data: ");
  Serial.println(tempData);

  // Convert to Celsius (0.25°C per LSB)
  float temperature = tempData * 0.25;

  return temperature;
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi Connection Failed!");
  }
}

void sendToGoogleSheets(float temp, int adc, float vout, float resistance,
                        float envTemp, float envHumidity) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    Serial.println("→ Sending data to Google Sheets...");

    // Create JSON payload
    String jsonData = "{";
    jsonData += "\"max6675_temp\":" + String(temp, 2) + ",";
    jsonData += "\"custom_adc\":" + String(adc) + ",";
    jsonData += "\"custom_vout\":" + String(vout, 3) + ",";
    jsonData += "\"custom_resistance\":" + String(resistance, 2) + ",";
    jsonData += "\"env_temp\":" + String(envTemp, 2) + ",";
    jsonData += "\"env_humidity\":" + String(envHumidity, 2);
    jsonData += "}";

    Serial.print("  JSON: ");
    Serial.println(jsonData);

    // Send POST request
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("  Response code: ");
      Serial.println(httpResponseCode);
      Serial.print("  Response: ");
      Serial.println(response);

      if (httpResponseCode == 200 || httpResponseCode == 302) {
        Serial.println("  ✓ Data sent successfully!");
      }
    } else {
      Serial.print("  ✗ Error: ");
      Serial.println(http.errorToString(httpResponseCode));
    }

    http.end();
  } else {
    Serial.println("  ✗ WiFi not connected!");
  }
}

////////// === BLYNK: push latest values to dashboard ===
void pushToBlynk() {
  // Only attempt if Blynk configured and WiFi available
  if (WiFi.status() != WL_CONNECTED) return;

  if (!Blynk.connected()) {
    // Try to reconnect quickly; Blynk.connect is nonblocking with short timeout here
    Blynk.connect(2000);
  }

  if (Blynk.connected()) {
    // If value is NAN, we send an empty string or a sentinel
    if (!isnan(g_Rsens)) Blynk.virtualWrite(V0, g_Rsens);
    else Blynk.virtualWrite(V0, "NaN");

    if (!isnan(g_envTemp)) Blynk.virtualWrite(V1, g_envTemp);
    else Blynk.virtualWrite(V1, "NaN");

    if (!isnan(g_envHumidity)) Blynk.virtualWrite(V2, g_envHumidity);
    else Blynk.virtualWrite(V2, "NaN");

    if (!isnan(g_thermoTemp)) Blynk.virtualWrite(V3, g_thermoTemp);
    else Blynk.virtualWrite(V3, "NaN");
  }
}
////////// === END BLYNK ADDED ===
