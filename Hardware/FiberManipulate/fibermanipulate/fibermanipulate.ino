#include "Arduino.h"
#include "teensystep4.h"
#include <iostream>
#include <string>
using namespace TS4;

int rotdisablepin = 23;
int movedisablepin = 22;

int dipperstep = 2;
int dipperdir = 3;

int stretcher1step = 0;
int stretcher1dir = 1;

int stretcher2step = 4;
int stretcher2dir = 5;

int rotator1step = 6;
int rotator1dir = 7;

int rotator2step = 8;
int rotator2dir = 9;

//Store which motors will do stuff
int dip, stret1, stret2, rot1, rot2;

//Holding the characters
const int maxStringLength = 64; // Maximum length of the string
char receivedString[maxStringLength]; // Array to store the received strin
char messageFromPC[2];
int motorselect;
int integerFromPC;
float floatFromPC;
bool newdata = false;





Stepper mdipper(dipperstep, dipperdir);
Stepper mstretch1(stretcher1step, stretcher1dir);
Stepper mstretch2(stretcher2step, stretcher2dir);
Stepper mrotator1(rotator1step, rotator1dir);
Stepper mrotator2(rotator2step, rotator2dir);

Stepper* allsteppers[] = { &mdipper, &mstretch1, &mstretch2, &mrotator1, &mrotator2 };

StepperGroup allmotors{mdipper,mstretch1,mstretch2,mrotator1,mrotator2};
StepperGroup rotators{mrotator1,mrotator2};
StepperGroup stretchers{mstretch1,mstretch2};
StepperGroup dipper{mdipper};
int whichmotors[5]; 
int motorcurrentpositions[5];

void setup()
{
  Serial.begin(9600);//Set the baud rate

  pinMode(rotdisablepin, OUTPUT);
  digitalWrite(rotdisablepin,LOW);
  pinMode(movedisablepin, OUTPUT);
  digitalWrite(movedisablepin,LOW);

    TS4::begin();

    //mdipper.setMaxSpeed(10'000);
    //mdipper.setAcceleration(50'000);

   // mdipper.moveAbs(-10000);
}


void loop() {
  String test = "";
  getString();
  if (newdata == true) {

        processString();
        newdata = false;
        performFunction();
    }
  
}

void getString(){
  
    // Check if there is any data available to read
    if (Serial.available() > 0) {
        // Read the incoming string
        int index = 0;
        while (index < maxStringLength - 1) { // Leave space for the null terminator
            if (Serial.available() > 0) {
                char incomingChar = Serial.read(); // Read a character
                if (incomingChar == '\n' || incomingChar == '\r') {
                    break; // Stop reading if a newline or carriage return is detected
                }
                receivedString[index++] = incomingChar; // Add character to the string
            }
        }
        receivedString[index] = '\0'; // Null-terminate the string

        // Print the received string
        newdata = true;
       
        // Optionally, you can clear the string for the next input
        //memset(receivedString, 0, sizeof(receivedString));
    }
    
}


void processString()
{
  char toprintstring[12];
  
  char * strtokIndx;
    strtokIndx = strtok(receivedString,",");      // get the first part - the string
    strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC
 
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    motorselect = atoi(strtokIndx);     // convert this part to an integer
    
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    integerFromPC = atoi(strtokIndx);     // convert this part to an integer

    strtokIndx = strtok(NULL, ",");
    floatFromPC = atof(strtokIndx);     // convert this part to a float
}

void performFunction() {
  //This code does the setting speeds etc and motions for the motors
  Serial.println(messageFromPC);
  //delay(10);
  //Set motor speeds
  if (strncmp("SS",messageFromPC,2) == 0) 
  {
    //delay(10);
      Serial.println("Setting Speeds");

    splitDigits(motorselect, dip, stret1, stret2, rot1, rot2);
    
    for (int i = 0; i < 5; i++) 
      {
      if (whichmotors[i]==1){
      allsteppers[i]->setMaxSpeed(floatFromPC);}
      }
  }
  //Set motor accelerations
  if (strncmp("SA",messageFromPC,2) == 0) 
  {
    //delay(10);
      Serial.println("Setting Accelerations");

    splitDigits(motorselect, dip, stret1, stret2, rot1, rot2);
    
    for (int i = 0; i < 5; i++) 
    {
      if (whichmotors[i]==1){
      allsteppers[i]->setAcceleration(floatFromPC); }
    }
  }

    //Set motor target relative
  if (strncmp("TR",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Setting relative targets");
    splitDigits(motorselect, dip, stret1, stret2, rot1, rot2);
    
    for (int i = 0; i < 5; i++) 
    {
      if (whichmotors[i]==1){
      motorcurrentpositions[i] = allsteppers[i]->getPosition();
      Serial.println(motorcurrentpositions[i]);
      allsteppers[i]->setTargetAbs(motorcurrentpositions[i]+integerFromPC);} 
    }
  }
  //Set motor target absolute
  if (strncmp("TA",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Setting absolute targets");
    splitDigits(motorselect, dip, stret1, stret2, rot1, rot2);
    
    for (int i = 0; i < 5; i++) 
    {
      if (whichmotors[i]==1){
      allsteppers[i]->setTargetAbs(integerFromPC); }
    }
  }
  //Make motors move
  if (strncmp("MA",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Moving motors");
     allmotors.move();
     
  }
    if (strncmp("MR",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Rotating");
     rotators.move();
     
  }
    if (strncmp("MD",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Moving dipper");
     dipper.move();
     
  }
      if (strncmp("MS",messageFromPC,2) == 0) 
  {
    //delay(10);
     Serial.println("Moving stetchers");
     stretchers.move();
     
  }
}


void splitDigits(int num, int &dip, int &stret1, int &stret2, int &rot1, int &rot2) {
    dip = num / 10000;             // Extract the first digit (ten thousand's place)
    stret1 = (num / 1000) % 10;      // Extract the second digit (thousand's place)
    stret2 = (num / 100) % 10;       // Extract the third digit (hundred's place)
    rot1 = (num / 10) % 10;        // Extract the fourth digit (ten's place)
    rot2 = num % 10;               // Extract the fifth digit (one's place)
    whichmotors[0] = dip;
    whichmotors[1] = stret1;
    whichmotors[2] = stret2;
    whichmotors[3] = rot1;
    whichmotors[4] = rot2;
}
