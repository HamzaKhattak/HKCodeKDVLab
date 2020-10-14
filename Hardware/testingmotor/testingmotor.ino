#include <AccelStepper.h>

//Select Pins
int Dir1Pin = 2; 
int Step1Pin = 3;

//Initiate the AccelStepper object
AccelStepper stepper(1,Step1Pin,Dir1Pin);

 
void setup()
{  
    Serial.begin(9600);
    Serial.println("This demo expects 3 pieces of data - text, an integer and a floating point value");
    Serial.println("Enter data in this style <HelloWorld, 12, 24.7>  ");
    Serial.println();
    stepper.setMaxSpeed(100);
    stepper.setAcceleration(10000);

}

//Holding the characters
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing

// variables to hold the parsed data
char messageFromPC[numChars] = {0};
int integerFromPC = 0;
float floatFromPC = 0.0;


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
    switch (messageFromPC[0]) {
      case 'S':
        stepper.setMaxSpeed(floatFromPC);
        stepper.setSpeed(floatFromPC);
        break;
        
      case 'M':
        stepper.moveTo(floatFromPC);
        while (stepper.currentPosition()!= floatFromPC)
          stepper.run();
          //Later add stops, checks etc here.
        break;
        
      case 'J':
        tempt = millis() + floatfromPC;
        //stepper.setSpeed(stepper.speed(floatFromPC)*intFromPC)
        while (millis()< tempt)
          {
          stepper.runSpeed()
          }
          //Later add stops, checks etc here.
        break;
        
      default:
        Serial.println('Invalid input')
      break;
    }
    Serial.print("Integer ");
    Serial.println(integerFromPC);
    Serial.print("Float ");
    Serial.println(floatFromPC);
}
