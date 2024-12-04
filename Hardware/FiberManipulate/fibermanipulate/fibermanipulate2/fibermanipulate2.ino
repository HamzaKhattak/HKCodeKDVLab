#include "TeensyStep.h"

//Select Pins
int Dir1Pin = 2; 
int Step1Pin = 3;

int Dir2Pin = 4; 
int Step2Pin = 5;

int Dir3Pin = 6; 
int Step3Pin = 7;

int Dir4Pin = 8; 
int Step4Pin = 9;
//Initiate the AccelStepper object
Stepper stepper1(Step1Pin,Dir1Pin);
Stepper stepper2(Step2Pin,Dir2Pin);
Stepper stepper3(Step3Pin,Dir3Pin);
Stepper stepper4(Step4Pin,Dir4Pin);
//StepperGroup gAll{stepper1,stepper2,stepper3,stepper4};
StepControl controller;
 
void setup()
{  
    Serial.begin(9600);//Set the baud rate
    stepper1.setMaxSpeed(100);
    stepper1.setAcceleration(10000);
    stepper2.setMaxSpeed(100);
    stepper2.setAcceleration(10000);
    stepper3.setMaxSpeed(100);
    stepper3.setAcceleration(10000);
    stepper4.setMaxSpeed(100);
    stepper4.setAcceleration(10000);

}

//Holding the characters
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing

// variables to hold the parsed data
char messageFromPC[numChars] = {0};
//String messageFromPC;
int integerFromPC = 0;
float floatFromPC = 0.0;
float tempt = 0;
float cur_speed = 0;
long cur_pos = 0;
boolean newData = false;



//============

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        strcpy(tempChars, receivedChars);
            // this temporary copy is necessary to protect the original data
            //   because strtok() used in parseData() replaces the commas with \0
        parseData();
        performFunction();
        newData = false;
    }
}

//============

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;
    int tempt = 0;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

//============

void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC
 
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    integerFromPC = atoi(strtokIndx);     // convert this part to an integer

    strtokIndx = strtok(NULL, ",");
    floatFromPC = atof(strtokIndx);     // convert this part to a float

}

//============

void performFunction() {
  Serial.println('we are here');
  if (strcmp('EB',messageFromPC) == 0) {
      Serial.println('we are here');
        stepper1.setTargetAbs(integerFromPC);
        stepper2.setTargetAbs(integerFromPC);
        stepper3.setTargetAbs(integerFromPC);
        stepper4.setTargetAbs(integerFromPC);
       
        controller.move(stepper1,stepper2,stepper3,stepper4);
          //Later add stops, checks etc here.
  }
    
}