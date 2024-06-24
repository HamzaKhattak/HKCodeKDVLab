/*==========================================================================
This sketch moves a feed motor back and forth and rotates a spindle motor to 
allow for a wire to be coiled around a bobbin
 *
 ===========================================================================*/

#include "TeensyStep.h"

Stepper spindlemotor(3, 4);       // STEP pin: 2, DIR pin: 3
Stepper feedmotor(6,7);
StepControl controller;    // Use default setting

float bobbinlength = 15 // length of the bobbin in mm
float wirediameter = 0.5 // diameter of the wire in mm



float leadpitch = 5.0; //mm per turn in the feed rail (ie pitch of lead screw)
int feedspr = 8000; //steps per rotation in feed motor
distperstep = mmperturn/feedmspr; //distance along bobbin per step of feed motor
feedstepsplayer = int(bobbinlength * feedspr / leadpitch) // Number of steps to traverse bobbin


int spindlespr = 8000; //steps per rotation in the spindle motor
turnsperbobbinL = bobbinlength/wirediameter
spindlestepsplayer = int(turnsperobbinL*spindlespr)


float feedspeed = 1 // speed the feeder moves in mm/s
feedsps = int(feedspeed * feedspr / leadpitch)// speed of feed motor in steps per second
spindlesps = (feedspeed/wirediameter)*spindlespr // speed of the spindle motor in steps per second


int timeacc = 3;



void setup()
{
     spindlemotor
  .setAcceleration(spindlesps/timeacc)
  .setMaxSpeed(spindlesps);
  

   feedmotor
  .setAcceleration(feedsps/timeacc)
  .setMaxSpeed(feedsps);

  delay(500);

    for (int i = 0; i <= 2; i++) {
    spindlemotor.setTargetRel(spindlestepsplayer);  // Set target position 
    feedmotor.setTargetRel(feedstepsplayer);  // Set target position 

    controller.move(spindlemotor,feedmotor);    // Do the move
    
    spindlemotor.setTargetRel(spindlestepsplayer);  // Set target position
    feedmotor.setTargetRel(-1*feedstepsplayer);  // Set target position
    controller.move(motor1,motor2);    // Do the move
    controller.move(spindlemotor,feedmotor);
  }
}


void loop() 
{

}