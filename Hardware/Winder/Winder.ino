/*==========================================================================
 * This is a minimal sketch showing the usage of TeensyStep
 *  
 * STEP Pulses on Pin 2    (can be any pin)
 * DIR  Signall on Pin 3   (can be any pin)
 * 
 * The target position is set to 1000 steps relative to the
 * current position. The move command of the controller 
 * moves the motor to the target position.  
 *  
 * Default parameters are 
 * Speed:          800 steps/s
 * Acceleration:  2500 steps/s^2
 * 
 * (slow, but good to start with since they will work on any normal stepper)
 *
 ===========================================================================*/

#include "TeensyStep.h"

Stepper spindlemotor(3, 4);       // STEP pin: 2, DIR pin: 3
Stepper feedmotor(6,7);
StepControl controller;    // Use default setting

float mmperturn = 5.0;
int feedmspr = 8000;
distperstep = mmperturn/feedmspr;

int spindlespr = 8000;

float bobbinlength = .03 // in m
float wirediameter = 0.5e-3

turnsperbobbinL = int(bobbinlength/wirediameter)-1
spindlestepsperlayer = turnsperbobbinL*



int m1speed = 60000;
int m2speed = 15000;
int timeacc = 1;
int timetotal = 5;
void setup()
{
     motor1
  .setAcceleration(m1speed/timeacc)
  .setMaxSpeed(m1speed);
  

   motor2
  .setAcceleration(m2speed/timeacc)
  .setMaxSpeed(m2speed);

  delay(500);

    for (int i = 0; i <= 1; i++) {
    motor1.setTargetRel(m1speed*timetotal);  // Set target position to 1000 steps from current position
    motor2.setTargetRel(m2speed*timetotal);  // Set target position to 1000 steps from current position

    controller.move(motor1,motor2);    // Do the move
    
    motor1.setTargetRel(m1speed*timetotal);  // Set target position to 1000 steps from current position
    motor2.setTargetRel(m2speed*-1*timetotal);  // Set target position to 1000 steps from current position
    controller.move(motor1,motor2);    // Do the move
  }
}


void loop() 
{

}