#include <Arduino.h>
#include "FastAccelStepper.h"
#include <Adafruit_NeoPixel.h>


#define enablePinStepper 5
#define dirPinStepper 3
#define stepPinStepper 1

#define switchpin 7
#define outputtestpin 9



//Motion direction and if moving
bool stopmotion = 1;
bool godirection = 0;
bool movetype = 0;


//Timing
uint64_t t0;
uint64_t t1;

//Stepper parameters
int currentpos = 0;
int destination = 0; 
float speed = 1.0;
float acceleration = 1.0;

//loop counters etc
int i;
bool movefinished = 1;
boolean newData = false;



char receivedChar;
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars]; // temporary array for use when parsing

// variables to hold the parsed data
char messageFromPC[numChars] = {0};
int integerFromPC = 0;
float floatFromPC = 0.0;


FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;

void recvOneChar()
{
  if (Serial.available() > 0)
  {
    receivedChar = Serial.read();
    newData = true;
  }
}
void showNewData()
{
  if (newData == true)
  {
    Serial.println(receivedChar);
    newData = false;
  }
}

void recvWithEndMarker()
{
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

  if (Serial.available() > 0)
  {
    rc = Serial.read();

    if (rc != endMarker)
    {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars)
      {
        ndx = numChars - 1;
      }
    }
    else
    {
      receivedChars[ndx] = '\0'; // terminate the string
      ndx = 0;
      newData = true;
    }
  }
}

void parseData() {      // split the data into its parts

    char * token; // this is used by strtok() as an index
    
    token = strtok(receivedChars,",");      // get the speed
    speed = atof(token);     

    token = strtok(NULL, ",");
    acceleration = atof(token);     // get acceleration

}


void setup()
{
  Serial.begin(115200);
  // Start the Serial Monitor at a baud rate of 115200
  pinMode(enablePinStepper, OUTPUT);
  digitalWrite(enablePinStepper, HIGH);
  pinMode(switchpin, INPUT_PULLUP);
  pinMode(outputtestpin, OUTPUT);

  engine.init();
  stepper = engine.stepperConnectToPin(stepPinStepper);
  stepper->setDirectionPin(dirPinStepper);
  stepper->setSpeedInHz(5000);     // 500 steps/s
  stepper->setAcceleration(50000); // 100 steps/s²
  // t1 = esp_timer_get_time();
  // t0 = esp_timer_get_time();
}

void loop()
{
  // t1 = esp_timer_get_time();
  // if ((t1-t0)>10e5){

  //  t0=t1;
  //}
  recvOneChar();
  if (newData)
  {
  switch (receivedChar)
  {

    case 's': //set speed and acceleration
    newData = false;
    while(newData == false){
    recvWithEndMarker();}
    parseData();
    stepper->setSpeedInHz(speed);     
    stepper->setAcceleration(acceleration);
    break;

    case 'f': //move forward continuously
    godirection = 1;
    stopmotion = 0;
    movetype = 1;
    break;

    case 'b': //move backwards continously 
    godirection = 0;
    stopmotion = 0;
    movetype = 1;
    break;

    case 'l': //get the position of the stepper motor
    currentpos = stepper->getCurrentPosition();
    Serial.println(currentpos);
    break;

    case 'm': //set destination speed and acceleration
    newData = false;
    while(newData == false){
    recvWithEndMarker();}
    destination = atoi(receivedChars);

    stepper->moveTo(destination);
    stopmotion = 0;
    movetype = 0; 
    movefinished = false;
    break;

    case 'h':
      stopmotion = 1;
    break;

    case 'g': //check if motion is done
      if(stepper->isRunning())
      {
        Serial.println("m");
      }
      else{
        Serial.println("d");
      }
      break;
    }
}
  if (!stopmotion)
  {
      if (movetype) // continuous or to destination
      {
        if (godirection == 1)
        {
          stepper->runForward();
        }
        else
        {
          stepper->runBackward();
        }
      }
      else
      {
        stepper->moveTo(destination);
        if (i%100 == 0)
        {
        if(!(stepper->isRunning()))
        {
          if (!movefinished)
          {
          movefinished = true;
          stopmotion = true;
          }
        }
        }
      }
  }

  
  else
  {
    if (stepper->isRunning())
    {
      stepper->stopMove();
    }
  }
  
  newData = false;
  i++;
}
