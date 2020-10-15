/*
Arduino code for stepper motor control
A bit ugly with global variables etc but works
Doesn't yet have end stops and other error catching
*/
#include <Arduino.h>
#include <AccelStepper.h>

//Select Pins
int Dir1Pin = 2; 
int Step1Pin = 3;

//Holding the characters
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing



// variables to hold the parsed data
char messageFromPC[numChars] = {0};
int integerFromPC = 0;
float floatFromPC = 0.0;
boolean newData = false;

// variables to temporary time position and speed
float tempt = 0;
float cur_speed = 0;
long cur_pos = 0;


//Initiate the AccelStepper object
AccelStepper stepper(1,Step1Pin,Dir1Pin);

//============

void recvWithStartEndMarkers() 
{
  /*
  As the main loop happens this function collects a string
  of form <char,int,float> with the < and > as end markers 
  */
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) 
      {
          rc = Serial.read();

          if (recvInProgress == true) 
            {
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
                    recvInProgress = false;
                    ndx = 0;
                    newData = true;
                  }
            }

          else if (rc == startMarker) {
              recvInProgress = true;
          }
      }
};

//============

void parseData() 

{      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC
 
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    integerFromPC = atoi(strtokIndx);     // convert this part to an integer

    strtokIndx = strtok(NULL, ",");
    floatFromPC = atof(strtokIndx);     // convert this part to a float

};

//============

void performFunction() 
{
    switch (messageFromPC[0]) {

      case 'G':
        cur_pos = stepper.currentPosition();
        Serial.println(cur_pos);
        break;
        
      case 'S':
        stepper.setMaxSpeed(floatFromPC);
        stepper.setSpeed(floatFromPC);
        break;

      case 'A':
        stepper.setAcceleration(floatFromPC);
        break;
        
      case 'M':
        stepper.moveTo(integerFromPC);
        while (stepper.currentPosition()!= integerFromPC)
          stepper.run();
          //Later add stops, checks etc here.
        break;
        
      case 'J':
        tempt = millis() + floatFromPC;
        //stepper.setSpeed(stepper.speed(floatFromPC)*intFromPC)
        while (millis() < tempt)
          {
          stepper.runSpeed();
          }
          //Later add stops, checks etc here.
        break;
        
      default:
        Serial.println("Invalid input");
      break;
    }
};


 
void setup()
{  
    Serial.begin(9600);
    stepper.setMaxSpeed(100);
    stepper.setAcceleration(1000);

};
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
};
