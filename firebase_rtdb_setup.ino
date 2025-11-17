#include <Wire.h>
#include "Adafruit_HTU21DF.h"
#include <WiFi.h>
#include <Firebase_ESP_Client.h>

// Insert your network credentials
#define WIFI_SSID "Hello"
#define WIFI_PASSWORD "123456789"

// Insert Firebase project API Key, URL and user credentials
#define API_KEY "API_KEY"
#define DATABASE_URL "https://temparature-monitor-92bef-default-rtdb.firebaseio.com" 

// Define Firebase Data object
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

Adafruit_HTU21DF htu = Adafruit_HTU21DF();

void setup() {
  Serial.begin(115200);

  // Initialize I2C Sensor
  if (!htu.begin()) {
    Serial.println("Couldn't find HTU21D sensor!");
    while (1);
  }

  // Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");

  // Setup Firebase
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  // Provide the user credentials for sign-in
  auth.user.email = "b22ee068@iitj.ac.in";     // Replace with your Firebase user email
  auth.user.password = "Password";        // Replace with your Firebase user password

  // Try to sign in
  if (Firebase.signUp(&config, &auth, auth.user.email, auth.user.password)) {
    Serial.println("Firebase sign-in successful");
  } else {
    Serial.printf("Sign-up error: %s\n", config.signer.signupError.message.c_str());
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  float temp = htu.readTemperature();
  float hum = htu.readHumidity();

  Serial.print("Temp: "); Serial.print(temp); Serial.print(" C ");
  Serial.print("Humidity: "); Serial.print(hum); Serial.println(" %");

  // Send data to Firebase
  if (Firebase.RTDB.setFloat(&fbdo, "/SensorData/temperature", temp)) {
    Serial.println("Temperature updated");
  } else {
    Serial.println("Failed to update temperature: " + fbdo.errorReason());
  }

  if (Firebase.RTDB.setFloat(&fbdo, "/SensorData/humidity", hum)) {
    Serial.println("Humidity updated");
  } else {
    Serial.println("Failed to update humidity: " + fbdo.errorReason());
  }

  delay(5000); // upload every 5 seconds
}
