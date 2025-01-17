#include <OPT3101.h>
#include <Wire.h>
#include <Pololu3piPlus2040.h>
#include <Arduino.h>
#include <SPI.h>
#include <U8g2lib.h>
#include <vector>

// include main library header file
#include <Chirale_TensorFlowLite.h>

// include static array definition of pre-trained model
#include "evo_2025-01-13T18-07-42_float32.h"

// This TensorFlow Lite Micro Library for Arduino is not similar to standard
// Arduino libraries. These additional header files must be included.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals pointers, used to address TensorFlow Lite components.
// Pointers are not usual in Arduino sketches, future versions of
// the library may change this...
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float my_x = 0.5;
float my_y = 0.5;

float goal_x = 9.5;
float goal_y = 9.5;

int action = 0;
float action_max = 0.0f;

bool driving = false;

// There is no way to calculate this parameter
// the value is usually determined by trial and errors
// It is the dimension of the memory area used by the TFLite interpreter
// to store tensors and intermediate results
constexpr int kTensorArenaSize = 3 * 1024;

// Keep aligned to 16 bytes for CMSIS (Cortex Microcontroller Software Interface Standard)
// alignas(16) directive is used to specify that the array
// should be stored in memory at an address that is a multiple of 16.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

OPT3101 sensor;
Motors motors;
ButtonA buttonA;
BumpSensors bumpSensors;
Buzzer buzzer;
Encoders encoders;
// OLED display;
LineSensors lineSensors;
RGBLEDs leds;
IMU imu;

U8G2_SH1106_128X64_NONAME_F_4W_SW_SPI u8g2(U8G2_R0, 2, /*data*/ 3, /* cs=*/U8X8_PIN_NONE, /* dc=*/0, /* reset=*/1);

// Allowed deviation (in degrees) relative to target angle
#define DEVIATION_THRESHOLD 3

#include "TurnSensor.h"

long countsRight = 0;
long prevRight = 0;
const int CLICKS_PER_ROTATION = 12;
const float GEAR_RATIO = 29.86F;
const int WHEEL_CIRCUMFERENCE = 10.0531;
float Sr = 0.0F;
int headingDirection = 0;
bool wallLeft = true;
bool wallRight = true;
int defaultSpeed = 80;

// method call to be able to define default parameters
void moveForward(bool forward = true, int count = 1);
void turn(char dir, int count = 1);

void setup() {
  // Initialize serial communications and wait for Serial Monitor to be opened
  Serial.begin(9600);
  u8g2.begin();
  u8g2.setFont(u8g2_font_ncenB14_tr);
  Wire.begin();
  delay(1000);
  
  Serial.println("Initializing TensorFlow Lite Micro Interpreter...");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);

  // Check if model and library have compatible schema version,
  // if not, there is a misalignement between TensorFlow version used
  // to train and generate the TFLite model and the current version of library
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided and schema version are not equal!");
    while (true)
      ;  // stop program here
  }

  // This pulls in all the TensorFlow Lite operators.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  // if an error occurs, stop the program.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true)
      ;  // stop program here
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  turnSensorSetup();
  turnSensorReset();

  sensor.init();
  if (sensor.getLastError()) {
    Serial.print(F("Failed to initialize OPT3101: error "));
    Serial.println(sensor.getLastError());
  }
  sensor.setFrameTiming(512);
  sensor.setBrightness(OPT3101Brightness::Adaptive);

  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  //buzzer.play("C32");

  Serial.println("Initialization done.");
  Serial.println("");
}


void loop() {
  motors.setSpeeds(0, 0);
  bumpSensors.read();
  if (buttonA.isPressed()) {
    delay(2000);
    driving = !driving;
  }
  if (driving) {

    delay(1000);

    input->data.f[0] = my_x;
    input->data.f[1] = my_y;

    input->data.f[2] = goal_x;
    input->data.f[3] = goal_y;

    for (int i = 0; i < 3; i++) {
      sensor.setChannel(i);
      sensor.sample();
      input->data.f[i + 4] = sensor.distanceMillimeters * 10;
    }
    for (int i = 0; i < 7; i++) {
      Serial.println(input->data.f[i]);
    }

    // Run inference, and report if an error occurs
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }
    action_max = 0.0f;
    u8g2.firstPage();
    for (int i = 0; i < 3; i++) {
      Serial.println(output->data.f[i]);
      u8g2.setCursor(0, (i + 1) * 15);
      switch (i) {
        case 0:
          u8g2.print("L: ");
          break;
        case 1:
          u8g2.print("R: ");
          break;
        case 2:
          u8g2.print("M: ");
          break;
      }
      u8g2.setCursor(30, (i + 1) * 15);
      u8g2.print(output->data.f[i]);
      if (output->data.f[i] > action_max) {
        action_max = output->data.f[i];
        action = i;
      }
    }
    u8g2.setCursor(0, 60);
    switch (action) {
      case 0:
        Serial.println("Turn Left");
        u8g2.print("Turn Left");
        u8g2.nextPage();
        turn('l', 1);
        break;
      case 1:
        Serial.println("Turn Right");
        u8g2.print("Turn Right");
        u8g2.nextPage();
        turn('r', 1);
        break;
      case 2:
        Serial.println("Move Forward");
        u8g2.print("Move Forward");
        u8g2.nextPage();
        moveForward(true, 1);
    }
  }
}

void turn(char dir, int count) {
  int turnSpeed = 80;
  int speed = 0;
  if (dir == 'l') {
    motors.setSpeeds(-turnSpeed, turnSpeed);
    headingDirection += 90 * count;
  } else if (dir == 'r') {
    motors.setSpeeds(turnSpeed, -turnSpeed);
    headingDirection -= 90 * count;
  }

  if (headingDirection < 0) {
    headingDirection += 360;
  } else if (headingDirection >= 360) {
    headingDirection -= 360;
  }

  while (true) {
    turnSensorUpdate();
    int32_t angle = getCurrentAngle();

    int diff = headingDirection - angle;
    int absDiff = abs(diff);
    if (absDiff > 100) {
      absDiff = 360 - absDiff;
    }

    speed = turnSpeed * absDiff / 180;
    speed += 25;

    if (dir == 'l') {
      motors.setSpeeds(-speed, speed);
    } else if (dir == 'r') {
      motors.setSpeeds(speed, -speed);
    }

    int threshPos = headingDirection + DEVIATION_THRESHOLD;
    if (threshPos < 0) {
      threshPos += 360;
    }

    int threshNeg = headingDirection - DEVIATION_THRESHOLD;
    if (threshPos < 0) {
      threshPos += 360;
    }

    if (angle <= threshPos && angle >= threshNeg) {
      motors.setSpeeds(0, 0);
      break;
    }
  }
}


void moveForward(bool forward, int count) {
  int speed = defaultSpeed;
  bool hasSampled = false;
  int distanceFront = 80;
  int distanceSide = 200;
  int driveDistance = 18 * count;
  wallRight = true;
  wallLeft = true;
  sensor.setChannel(1);
  sensor.sample();
  if (forward && sensor.distanceMillimeters < distanceFront) {
    return;
  }

  for (int i = 0; i < 3; i++) {
    sensor.setChannel(i);
    sensor.startSample();
  }

  Sr = 0.0F;
  countsRight = encoders.getCountsAndResetRight();
  countsRight = 0;
  prevRight = 0;

  while (true) {
    turnSensorUpdate();

    sensor.nextChannel();
    // start sample evaluation shortly after moving to avoid detecting an opening at the start
    if (sensor.isSampleDone() && Sr > 2) {
      sensor.readOutputRegs();
      switch (sensor.channelUsed) {
        case 0:
          if (wallLeft && sensor.distanceMillimeters > distanceSide) {
            wallLeft = false;
          }
          break;
        case 1:
          if (sensor.distanceMillimeters < distanceFront) {
            goto bailout;
          }
          break;
        case 2:
          if (wallRight && sensor.distanceMillimeters > distanceSide) {
            wallRight = false;
          }
          break;
      }
      sensor.startSample();
    }

    countsRight += encoders.getCountsAndResetRight();

    Sr += ((countsRight - prevRight) / (CLICKS_PER_ROTATION * GEAR_RATIO) * WHEEL_CIRCUMFERENCE);

    int diff = headingDirection - getCurrentAngle();
    int absDiff = abs(diff);
    if (absDiff > 100) {
      absDiff = 360 - absDiff;
    }

    int turnSpeed = speed * absDiff / 20;
    if (turnSpeed > 10) {
      turnSpeed = 10;
    }

    if (Sr < driveDistance && Sr > -driveDistance) {
      if (Sr > driveDistance - 5 || Sr < -driveDistance + 5) {
        speed = defaultSpeed - (abs(Sr) * 3);
      }
      if (speed < 25) {
        speed = 25;
      }
      if (diff > 0 || diff < -270) {
        forward ? motors.setSpeeds(speed, speed + turnSpeed) : motors.setSpeeds(-speed - turnSpeed, -speed);
      } else if (diff < 0) {
        forward ? motors.setSpeeds(speed + turnSpeed, speed) : motors.setSpeeds(-speed, -speed - turnSpeed);
      } else {
        forward ? motors.setSpeeds(speed, speed) : motors.setSpeeds(-speed, -speed);
      }
    } else {
      break;
    }

    prevRight = countsRight;
  }

bailout:
  motors.setSpeeds(0, 0);
  switch (headingDirection) {
    case 0:
      my_y += 0.5;
      break;
    case 90:
      my_x -= 0.5;
      break;
    case 180:
      my_y -= 0.5;
      break;
    case 270:
      my_x += 0.5;
      break;
  }
}

int32_t getCurrentAngle() {
  int32_t angle = (((int32_t)turnAngle >> 16) * 360) >> 16;
  if (angle < 0) {
    angle += 360;
  }

  return angle;
}
