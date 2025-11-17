#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_HTU21DF.h>

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
  
  Serial.println("Setup complete!\n");
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Reconnecting...");
    connectWiFi();
  }
  
  unsigned long currentTime = millis();
  
  // Read sensors at defined interval
  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;
    
    // --- Read MAX6675 ---
    float thermoTemp = readMAX6675();
    
    // --- Read Custom Sensor ---
    int adcVal = analogRead(adcPin);
    float Vout = (adcVal * Vcc) / adcMax;
    float Rsens = 0;
    
    if (adcVal > 0 && adcVal < adcMax) {
      Rsens = Rref * ((Vcc / Vout) - 1.0);
    }
    
    // --- Read HTU21D (Environment) ---
    float envTemp = htu.readTemperature();
    float envHumidity = htu.readHumidity();
    
    // --- Display readings ---
    Serial.println("========== Sensor Readings ==========");
    
    // MAX6675
    if (isnan(thermoTemp)) {
      Serial.println("MAX6675: No thermocouple detected!");
      thermoTemp = -999; 
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
      envTemp = -999;
      envHumidity = -999;
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