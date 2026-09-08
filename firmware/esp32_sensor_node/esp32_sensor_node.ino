/* ===========================================================================
 * Crane Operator Fatigue Monitor - cabin sensor node
 *
 * Target : ESP32 DevKit v1 (also builds for ESP8266 / Arduino Nano with the
 *          I2C pins changed)
 * Role   : read the MAX30100 pulse oximeter and the MPU6050 IMU, drive the
 *          local LED / buzzer / vibration annunciators, and stream one JSON
 *          line per sample to the edge unit over USB serial.
 *
 * Why a separate microcontroller rather than wiring the sensors straight to
 * the Raspberry Pi: the MAX30100 needs its beat interrupt serviced with tight
 * timing, and a Pi running OpenCV at 15 fps under Linux cannot guarantee that.
 * Putting the sampling on a dedicated MCU keeps the beat detection accurate
 * and leaves the Pi free for the vision pipeline.
 *
 * Wiring (ESP32 DevKit v1)
 *   MAX30100  SDA -> GPIO21    SCL -> GPIO22   VIN -> 3V3   GND -> GND
 *   MPU6050   SDA -> GPIO21    SCL -> GPIO22   VCC -> 3V3   GND -> GND
 *             (both share the I2C bus; addresses 0x57 and 0x68 do not clash)
 *   Buzzer         -> GPIO23   (through a 2N2222 if using a 12 mm piezo)
 *   Vibration motor-> GPIO19   (through a transistor + flyback diode)
 *   LED green      -> GPIO25   (each via a 220R resistor)
 *   LED yellow     -> GPIO26
 *   LED red        -> GPIO27
 *
 * Libraries (Arduino IDE -> Library Manager)
 *   "MAX30100lib"      by OXullo Intersecans
 *   "Adafruit MPU6050" + "Adafruit Unified Sensor"
 *   "ArduinoJson"      by Benoit Blanchon
 *
 * Serial protocol
 *   Up   (this node -> edge unit), one JSON object per line at 20 Hz:
 *     {"hr":72.4,"spo2":97.1,"beat":0,"ax":0.01,"ay":-0.03,"az":0.99,
 *      "gx":0.4,"gy":-1.1,"gz":0.2,"t":128374}
 *   Down (edge unit -> this node), one command per line:
 *     STATE:NORMAL | STATE:WARNING | STATE:CRITICAL | PING
 *
 * The edge unit works fine without this board - edge/sensors.py falls back to
 * simulation - so the vision half of the system can be demonstrated before the
 * hardware is assembled.
 * ===========================================================================
 */

#include <Wire.h>
#include <ArduinoJson.h>
#include "MAX30100_PulseOximeter.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ---------------------------------------------------------------- pin map
static const uint8_t PIN_BUZZER    = 23;
static const uint8_t PIN_VIBRATION = 19;
static const uint8_t PIN_LED_GREEN = 25;
static const uint8_t PIN_LED_AMBER = 26;
static const uint8_t PIN_LED_RED   = 27;

// ---------------------------------------------------------------- timing
static const uint32_t SAMPLE_PERIOD_MS = 50;    // 20 Hz telemetry
static const uint32_t LINK_TIMEOUT_MS  = 5000;  // edge unit considered gone

// ESP32 LEDC channel used to drive the piezo buzzer.
static const uint8_t  BUZZER_CHANNEL = 0;
static const uint16_t BUZZER_FREQ_WARN = 2000;
static const uint16_t BUZZER_FREQ_CRIT = 3200;

// ---------------------------------------------------------------- state
enum AlertState { STATE_NORMAL, STATE_WARNING, STATE_CRITICAL };

PulseOximeter pox;
Adafruit_MPU6050 mpu;

bool     mpuReady      = false;
bool     poxReady      = false;
bool     beatFlag      = false;      // set by the ISR-style callback
uint32_t lastSample    = 0;
uint32_t lastCommand   = 0;
uint32_t patternClock  = 0;
bool     patternOn     = false;
AlertState alertState  = STATE_NORMAL;

/* Called by the MAX30100 library on every detected beat. Keep it trivial -
 * the heavy lifting (RR intervals, HRV) happens on the edge unit, which has
 * floating point to spare. */
void onBeatDetected() {
  beatFlag = true;
}

// ------------------------------------------------------------- annunciators

void setLeds(bool green, bool amber, bool red) {
  digitalWrite(PIN_LED_GREEN, green);
  digitalWrite(PIN_LED_AMBER, amber);
  digitalWrite(PIN_LED_RED,   red);
}

void buzzerOn(uint16_t frequency, uint8_t dutyPercent) {
  ledcWriteTone(BUZZER_CHANNEL, frequency);
  ledcWrite(BUZZER_CHANNEL, map(dutyPercent, 0, 100, 0, 255));
}

void buzzerOff() {
  ledcWrite(BUZZER_CHANNEL, 0);
}

void applyAlertState() {
  switch (alertState) {
    case STATE_NORMAL:
      setLeds(true, false, false);
      buzzerOff();
      digitalWrite(PIN_VIBRATION, LOW);
      break;
    case STATE_WARNING:
      setLeds(false, true, false);
      digitalWrite(PIN_VIBRATION, LOW);
      break;
    case STATE_CRITICAL:
      setLeds(false, false, true);
      break;
  }
}

/* Non-blocking beep patterns. delay() must never be used here: it would stall
 * pox.update(), and the MAX30100 loses beat detection if it is not polled
 * continuously. */
void serviceAlertPattern(uint32_t now) {
  uint32_t onMs, offMs;
  switch (alertState) {
    case STATE_WARNING:  onMs = 180; offMs = 1200; break;
    case STATE_CRITICAL: onMs = 400; offMs = 200;  break;
    default:
      buzzerOff();
      return;
  }

  uint32_t interval = patternOn ? onMs : offMs;
  if (now - patternClock < interval) return;
  patternClock = now;
  patternOn = !patternOn;

  if (patternOn) {
    buzzerOn(alertState == STATE_CRITICAL ? BUZZER_FREQ_CRIT : BUZZER_FREQ_WARN,
             alertState == STATE_CRITICAL ? 50 : 22);
    if (alertState == STATE_CRITICAL) digitalWrite(PIN_VIBRATION, HIGH);
  } else {
    buzzerOff();
    digitalWrite(PIN_VIBRATION, LOW);
  }
}

// ---------------------------------------------------------------- downlink

void readCommands(uint32_t now) {
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    lastCommand = now;

    if (line == "PING") {
      Serial.println("{\"pong\":1}");
    } else if (line.startsWith("STATE:")) {
      String value = line.substring(6);
      if      (value == "NORMAL")   alertState = STATE_NORMAL;
      else if (value == "WARNING")  alertState = STATE_WARNING;
      else if (value == "CRITICAL") alertState = STATE_CRITICAL;
      patternClock = now;
      patternOn = false;
      applyAlertState();
    }
  }

  /* Fail safe: if the edge unit stops talking to us, the operator has no
   * monitoring at all. Hold the red LED so the loss is visible in the cabin
   * rather than failing silently to green. */
  if (lastCommand != 0 && (now - lastCommand) > LINK_TIMEOUT_MS
      && alertState == STATE_NORMAL) {
    setLeds(false, false, true);
  }
}

// ------------------------------------------------------------------- setup

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_AMBER, OUTPUT);
  pinMode(PIN_LED_RED,   OUTPUT);
  pinMode(PIN_VIBRATION, OUTPUT);

  ledcSetup(BUZZER_CHANNEL, BUZZER_FREQ_WARN, 8);
  ledcAttachPin(PIN_BUZZER, BUZZER_CHANNEL);
  buzzerOff();

  // Lamp test, so a dead LED is caught before the shift rather than during it.
  setLeds(true, true, true);
  delay(500);
  setLeds(false, false, false);

  Wire.begin(21, 22);

  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    // 21 Hz bandwidth: well above head-nod frequencies, far below the cabin
    // vibration that would otherwise dominate the signal on a working crane.
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    mpuReady = true;
  }

  if (pox.begin()) {
    pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
    pox.setOnBeatDetectedCallback(onBeatDetected);
    poxReady = true;
  }

  StaticJsonDocument<128> boot;
  boot["boot"] = 1;
  boot["mpu6050"]  = mpuReady;
  boot["max30100"] = poxReady;
  serializeJson(boot, Serial);
  Serial.println();

  applyAlertState();
  lastSample = millis();
}

// -------------------------------------------------------------------- loop

void loop() {
  // Must run on every pass or beat detection degrades.
  if (poxReady) pox.update();

  uint32_t now = millis();
  readCommands(now);
  serviceAlertPattern(now);

  if (now - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample = now;

  StaticJsonDocument<256> doc;

  if (poxReady) {
    doc["hr"]   = pox.getHeartRate();
    doc["spo2"] = pox.getSpO2();
  }
  doc["beat"] = beatFlag ? 1 : 0;
  beatFlag = false;

  if (mpuReady) {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    // Report acceleration in g and rotation in deg/s, which is what
    // edge/sensors.py::tilt_from_accel expects.
    doc["ax"] = accel.acceleration.x / 9.80665f;
    doc["ay"] = accel.acceleration.y / 9.80665f;
    doc["az"] = accel.acceleration.z / 9.80665f;
    doc["gx"] = gyro.gyro.x * 57.2958f;
    doc["gy"] = gyro.gyro.y * 57.2958f;
    doc["gz"] = gyro.gyro.z * 57.2958f;
  }

  doc["t"] = now;

  serializeJson(doc, Serial);
  Serial.println();
}
