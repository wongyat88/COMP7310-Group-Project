#include <esp_camera.h>
#include <WiFi.h>
#include <PubSubClient.h>

// ------ WiFi Credentials ------
const char* ssid = "gateWayNet";
const char* password = "Fuwamoco2023";

// ------ MQTT broker ------
const char* mqtt_server = "192.168.137.1";  // Windows AP, modify to your MQTT broker IP address
const unsigned int mqtt_port = 1883;        // MQTT broker port
#define MQTT_PUBLISH_Monitor "TESTING2"     // MQTT topic for publishing monitor image

// ------ OV2640 camera configuration ------------
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 21
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 19
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 5
#define Y2_GPIO_NUM 4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

uint32_t Freq = 0;
char clientId[50];
void mqtt_callback(char* topic, byte* payload, unsigned int msgLength);
WiFiClient wifiClient;
PubSubClient mqttClient(mqtt_server, mqtt_port, mqtt_callback, wifiClient);
const unsigned int desiredFPS = 24;                      // Set the desired frames per second
const unsigned int captureInterval = 1000 / desiredFPS;  // Set the desired capture interval in milliseconds
int counter = 0;                                         // Counter variable

// Wi-Fi Connection
void setup_wifi() {
  Serial.printf("\nConnecting to %s", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("\nWiFi Connected.  IP Address: ");
  Serial.println(WiFi.localIP());
}

// MQTT callback
void mqtt_callback(char* topic, byte* payload, unsigned int msgLength) {
}

// MQTT reconnect, for non blocking MQTT client
boolean mqtt_nonblock_reconnect() {
  boolean doConn = false;
  if (!mqttClient.connected()) {
    boolean isConn = mqttClient.connect(clientId);
    char logConnected[100];
    sprintf(logConnected, "MQTT Client [%s] Connect %s !", clientId, (isConn ? "Successful" : "Failed"));
    Serial.println(logConnected);
  }
  return doConn;
}

// Publish picture to MQTT broker
void MQTT_picture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed, Reset");
    ESP.restart();
  }

  if (!mqttClient.connected()) {
    Serial.printf("MQTT Client Connection Lost\n");
    mqtt_nonblock_reconnect();
  }

  if (mqttClient.connected()) {
    int imgSize = fb->len;
    int ps = MQTT_MAX_PACKET_SIZE;
    mqttClient.beginPublish(MQTT_PUBLISH_Monitor, imgSize, false);
    for (int i = 0; i < imgSize; i += ps) {
      int s = (imgSize - i < ps) ? (imgSize - i) : ps;
      mqttClient.write((uint8_t*)(fb->buf) + i, s);
    }
    boolean isPublished = mqttClient.endPublish();
    if (isPublished) {
      Serial.print("Photo published to MQTT successfully. Counter: ");
      Serial.println(counter);
    } else {
      Serial.print("Failed to publish photo to MQTT. Counter: ");
      Serial.println(counter);
    }
    counter++;
  }

  esp_camera_fb_return(fb);
}

// Main function
void setup() {
  // Serial port for debugging purposes
  Serial.begin(115200);

  // Configure processor frequency
  setCpuFrequencyMhz(240);

  // Camera Config
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 28000000;
  config.pixel_format = PIXFORMAT_JPEG;  // PIXFORMAT_YUV422,PIXFORMAT_GRAYSCALE,PIXFORMAT_RGB565,PIXFORMAT_JPEG
  config.jpeg_quality = 10;              //10-63 lower number means higher quality
  config.fb_count = 1;
  config.frame_size = FRAMESIZE_QVGA;  // FRAMESIZE_ + UXGA|SXGA|XGA|SVGA|VGA|CIF|QVGA|HQVGA|QQVGA
  esp_err_t err = esp_camera_init(&config);
  setup_wifi();
  sprintf(clientId, "ESP32CAM_%04X", random(0xffff));  // Create a random client ID
  mqtt_nonblock_reconnect();
}

void loop() {
  MQTT_picture();         // Publish picture to MQTT broker
}