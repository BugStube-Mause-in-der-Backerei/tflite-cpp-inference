#include <OPT3101.h>
#include <Wire.h>
#include <Pololu3piPlus2040.h>
#include <Arduino.h>
#include <SPI.h>
#include <U8g2lib.h>
#include <vector>

U8G2_SH1106_128X64_NONAME_F_4W_SW_SPI u8g2(U8G2_R0, 2, 3, U8X8_PIN_NONE, 0, 1);

#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(x) Serial.print(x)
#define DEBUG_PRINTF(x, y) Serial.print(x, y)
#define DEBUG_PRINTLN(x) Serial.println(x)
#else
#define DEBUG_PRINT(x)
#define DEBUG_PRINTF(x, y)
#define DEBUG_PRINTLN(x)
#endif

// Allowed deviation (in degrees) relative to target angle
#define DEVIATION_THRESHOLD 3

OPT3101 sensor;
Buzzer buzzer;
ButtonA buttonA;
ButtonB buttonB;
ButtonC buttonC;
LineSensors lineSensors;
BumpSensors bumpSensors;
Motors motors;
Encoders encoders;
RGBLEDs leds;
IMU imu;

#include "TurnSensor.h"

const int WHEEL_CIRCUMFERENZCE = 10.0531;
const int CLICKS_PER_ROTATION = 12;
const float GEAR_RATIO = 29.86F;
bool wallLeft = true;
bool wallRight = true;
bool startingWall = false;
long countsRight = 0;
long prevRight = 0;
float distanceRight = 0.0F;
int headingDirection = 0;
int defaultSpeed = 80;
int currentPos[] = { 0, 0 };
int endPos[] = { 2, 1 };
String endProgramm = "";

// method call to be able to define default parameters
bool moveForward(bool forward = true, int count = 1);
void turn(char dir, int count = 1);

void pretty_print_vector(const std::vector<std::tuple<float, float>>& vec) {
  DEBUG_PRINTLN("[");
  for (size_t i = 0; i < vec.size(); ++i) {
    const auto& [x, y] = vec[i];
    DEBUG_PRINT("[");
    DEBUG_PRINTF(x, 2);
    DEBUG_PRINT(", ");
    DEBUG_PRINTF(y, 2);
    DEBUG_PRINT("]");
    if (i < vec.size() - 1) DEBUG_PRINTLN(",");
  }
  DEBUG_PRINTLN("");
  DEBUG_PRINTLN("]");
}

// TF-Lite stuff
#include <Chirale_TensorFlowLite.h>

// include static array definition of pre-trained model
#include "dql_2025-01-16T19-45-36_float32.h"

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

int action = 0;
float action_max = 0.0f;

enum actions {
  LEFT = 0,
  RIGHT = 1,
  DOWN = 2,
  UP = 3
};

enum Direction {
  NORTH = 0,
  EAST = 270,
  SOUTH = 180,
  WEST = 90
};

std::vector<std::tuple<float, float>> vec;

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


void setup() {
  // Initialize serial communications and wait for Serial Monitor to be opened
  Serial.begin(9600);
  u8g2.begin();
  u8g2.setFont(u8g2_font_ncenB14_tr);
  Wire.begin();
  delay(1000);
  for (int i = 0; i < 10; i++) {
    std::tuple<float, float> start(.5f, .5f);
    vec.push_back(start);
  }

  pretty_print_vector(vec);

  randomSeed(5);

  turnSensorSetup();
  turnSensorReset();

  sensor.init();
  if (sensor.getLastError()) {
    DEBUG_PRINT(F("Failed to initialize OPT3101: error "));
    DEBUG_PRINTLN(sensor.getLastError());
  }
  sensor.setFrameTiming(512);
  sensor.setBrightness(OPT3101Brightness::Adaptive);

  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  DEBUG_PRINTLN("Initializing TensorFlow Lite Micro Interpreter...");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);

  // Check if model and library have compatible schema version,
  // if not, there is a misalignement between TensorFlow version used
  // to train and generate the TFLite model and the current version of library
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    DEBUG_PRINTLN("Model provided and schema version are not equal!");
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
    DEBUG_PRINTLN("AllocateTensors() failed");
    while (true)
      ;  // stop program here
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

#ifdef DEBUG
  // Ensure the input and output tensors have the correct shapes
  DEBUG_PRINT("nullptr != input: ");
  DEBUG_PRINTLN(nullptr != input ? "True" : "False");
  DEBUG_PRINT("3 == input->dims->size: ");
  DEBUG_PRINTLN(3 == input->dims->size ? "True" : "False");
  DEBUG_PRINT("1 == input->dims->data[0]: ");
  DEBUG_PRINTLN(1 == input->dims->data[0] ? "True" : "False");
  DEBUG_PRINT("2 == input->dims->data[1]: ");
  DEBUG_PRINTLN(2 == input->dims->data[1] ? "True" : "False");
  DEBUG_PRINT("10 == input->dims->data[2]: ");
  DEBUG_PRINTLN(10 == input->dims->data[2] ? "True" : "False");
  DEBUG_PRINT("kTfLiteFloat32 == input->type: ");
  DEBUG_PRINTLN(kTfLiteFloat32 == input->type ? "True" : "False");

  DEBUG_PRINT("nullptr != output: ");
  DEBUG_PRINTLN(nullptr != output ? "True" : "False");
  DEBUG_PRINT("2 == output->dims->size: ");
  DEBUG_PRINTLN(2 == output->dims->size ? "True" : "False");
  DEBUG_PRINT("1 == output->dims->data[0]: ");
  DEBUG_PRINTLN(1 == output->dims->data[0] ? "True" : "False");
  DEBUG_PRINT("4 == output->dims->data[1]: ");
  DEBUG_PRINTLN(4 == output->dims->data[1] ? "True" : "False");
  DEBUG_PRINT("kTfLiteFloat32 == output->type: ");
  DEBUG_PRINTLN(kTfLiteFloat32 == output->type ? "True" : "False");
#endif

  DEBUG_PRINTLN(NORTH);
  DEBUG_PRINTLN(EAST);
  DEBUG_PRINTLN(SOUTH);
  DEBUG_PRINTLN(WEST);

  buzzer.play("C32");

  DEBUG_PRINTLN("Initialization done.");
  DEBUG_PRINTLN("");
}


void loop() {
  motors.setSpeeds(0, 0);
  bumpSensors.read();
  if (buttonA.isPressed()) {
    delay(2000);
    driving = !driving;
  }
  if (driving) {

    delay(500);
    // fill input tensor of the model
    for (int i = 0; i < vec.size(); i++) {
      for (int j = 0; j < 2; j++) {
        if (j == 0) {
          DEBUG_PRINTLN(std::get<0>(vec[i]));
          input->data.f[i * 2 + j] = std::get<0>(vec[i]);
        } else {
          DEBUG_PRINTLN(std::get<1>(vec[i]));
          input->data.f[i * 2 + j] = std::get<1>(vec[i]);
        }
      }
    }

    // Run inference, and report if an error occurs
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      DEBUG_PRINTLN("Invoke failed!");
      return;
    }

    action_max = output->data.f[0];
    u8g2.firstPage();
    for (int i = 0; i < 4; i++) {
      DEBUG_PRINTLN(output->data.f[i]);
      if (output->data.f[i] > action_max) {
        action_max = output->data.f[i];
        action = i;
        DEBUG_PRINTLN(action);
      }
    }
    u8g2.setCursor(0, 60);
    switch (action) {
      case LEFT:
        DEBUG_PRINTLN("Move Left");
        u8g2.print("Move Left");
        u8g2.nextPage();
        move('l');
        break;
      case RIGHT:
        DEBUG_PRINTLN("Move Right");
        u8g2.print("Move Right");
        u8g2.nextPage();
        move('r');
        break;
      case UP:
        DEBUG_PRINTLN("Move Up");
        u8g2.print("Move Up");
        u8g2.nextPage();
        move('u');
        break;
      case DOWN:
        DEBUG_PRINTLN("Move Down");
        u8g2.print("Move Down");
        u8g2.nextPage();
        move('d');
        break;
    }
    vec.erase(vec.begin());
    std::tuple<float, float> new_pos(currentPos[0], currentPos[1]);
    vec.push_back(new_pos);
  }
}

void move(char dir) {
  switch (dir) {
    case 'u':
      switch (headingDirection) {
        case NORTH:
          if (moveForward()) updateCurrentPos(true);
          break;
        case EAST:
          turn('l');
          if (moveForward()) updateCurrentPos(true);
          break;
        case SOUTH:
          turn('l', 2);
          if (moveForward()) updateCurrentPos(true);
          break;
        case WEST:
          turn('r');
          if (moveForward()) updateCurrentPos(true);
          break;
      }
      break;
    case 'd':
      switch (headingDirection) {
        case NORTH:
          turn('l', 2);
          if (moveForward()) updateCurrentPos(true);
          break;
        case EAST:
          turn('r');
          if (moveForward()) updateCurrentPos(true);
          break;
        case SOUTH:
          if (moveForward()) updateCurrentPos(true);
          break;
        case WEST:
          turn('l');
          if (moveForward()) updateCurrentPos(true);
          break;
      }
      break;
    case 'l':
      switch (headingDirection) {
        case NORTH:
          turn('l');
          if (moveForward()) updateCurrentPos(true);
          break;
        case EAST:
          turn('l', 2);
          if (moveForward()) updateCurrentPos(true);
          break;
        case SOUTH:
          turn('r');
          if (moveForward()) updateCurrentPos(true);
          break;
        case WEST:
          if (moveForward()) updateCurrentPos(true);
          break;
      }
      break;
    case 'r':
      switch (headingDirection) {
        case NORTH:
          turn('r');
          if (moveForward()) updateCurrentPos(true);
          break;
        case EAST:
          if (moveForward()) updateCurrentPos(true);
          break;
        case SOUTH:
          turn('l');
          if (moveForward()) updateCurrentPos(true);
          break;
        case WEST:
          turn('l', 2);
          if (moveForward()) updateCurrentPos(true);
          break;
      }
      break;
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


bool moveForward(bool forward, int count) {
  int speed = defaultSpeed;
  int speedLeft = 0;
  int speedRight = 0;
  int distanceFront = 80;
  int distanceSide = 200;
  bool hasSampled = false;
  float driveDistance = 17.25F * count;
  wallRight = true;
  wallLeft = true;
  sensor.setChannel(1);
  sensor.startSample();
  while (!sensor.isSampleDone()) {}
  sensor.readOutputRegs();

  if (forward && sensor.distanceMillimeters < distanceFront) {
    return false;
  }

  for (int i = 0; i < 3; i++) {
    sensor.setChannel(i);
    sensor.startSample();
  }

  distanceRight = 0.0F;
  countsRight = encoders.getCountsAndResetRight();
  countsRight = 0;
  prevRight = 0;

  while (true) {
    turnSensorUpdate();

    sensor.nextChannel();
    // start sample evaluation shortly after moving to avoid detecting an opening at the start
    if (sensor.isSampleDone() && distanceRight >= 2) {
      sensor.readOutputRegs();
      int distanceMM = sensor.distanceMillimeters;
      // recognize wall openings during moving
      switch (sensor.channelUsed) {
        case 0:
          if (wallLeft && distanceMM > distanceSide) {
            wallLeft = false;
          }
          if (distanceMM < 150) {
            speedLeft = 10;
          } else {
            speedLeft = 0;
          }
          break;
        case 1:
          if (distanceMM < distanceFront) {
            goto bailout;
          }
          break;
        case 2:
          if (wallRight && distanceMM > distanceSide) {
            wallRight = false;
          }
          if (distanceMM < 150) {
            speedRight = 10;
          } else {
            speedRight = 0;
          }
          break;
      }
      sensor.startSample();
    }


    countsRight += encoders.getCountsAndResetRight();
    // approximate the driven distance
    distanceRight += ((countsRight - prevRight) / (CLICKS_PER_ROTATION * GEAR_RATIO) * WHEEL_CIRCUMFERENZCE);

    int diff = headingDirection - getCurrentAngle();
    int absDiff = abs(diff);
    // diff shouldn't be negative; edge case: headingDirection = 0 and angle = 350 should be diff 10 and not 350
    if (absDiff > 100) {
      absDiff = 360 - absDiff;
    }

    // turnSpeed shouldn't be greater than 10 as the robot would start to wiggle
    int turnSpeed = speed * absDiff / 20;
    if (turnSpeed > 10) {
      turnSpeed = 10;
    } else if (speedLeft != 0 || speedRight != 0) {
      turnSpeed = 0;
    }

    if (distanceRight <= driveDistance && distanceRight >= -driveDistance) {
      if (distanceRight > driveDistance - 5 || distanceRight < -driveDistance + 5) {
        speed = defaultSpeed - (abs(distanceRight) * 3);
      }
      if (speed < 25) {
        speed = 25;
      }

      if (diff > 0 || diff < -270) {
        forward ? motors.setSpeeds(speed + speedLeft, speed + speedRight + turnSpeed) : motors.setSpeeds(-speed - turnSpeed, -speed);
      } else if (diff < 0) {
        forward ? motors.setSpeeds(speed + speedLeft + turnSpeed, speed + speedRight) : motors.setSpeeds(-speed, -speed - turnSpeed);
      } else {
        forward ? motors.setSpeeds(speed + speedLeft, speed + speedRight) : motors.setSpeeds(-speed, -speed);
      }
    } else {
      break;
    }

    prevRight = countsRight;
  }

bailout:
  motors.setSpeeds(0, 0);
  DEBUG_PRINTLN(distanceRight);
  return distanceRight > 12.0F ? true : false;
}

void updateCurrentPos(bool movedForward) {
  switch (headingDirection) {
    case 0:
      movedForward ? changeCurrentPos(0, 1) : changeCurrentPos(0, -1);
      break;
    case 90:
      movedForward ? changeCurrentPos(1, 1) : changeCurrentPos(1, -1);
      break;
    case 180:
      movedForward ? changeCurrentPos(0, -1) : changeCurrentPos(0, 1);
      break;
    case 270:
      movedForward ? changeCurrentPos(1, -1) : changeCurrentPos(1, 1);
      break;
  }
}

void changeCurrentPos(int field, int change) {
  currentPos[field] = currentPos[field] + change;
  Serial.println(currentPos[field]);
}

int32_t getCurrentAngle() {
  int32_t angle = (((int32_t)turnAngle >> 16) * 360) >> 16;
  if (angle < 0) {
    angle += 360;
  }

  return angle;
}
