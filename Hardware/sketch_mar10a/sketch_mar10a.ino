//initialposition
const int trigPin = 9;
cont in echoPin = 10;
long duration;
long distance;

int analogPin = 3

bool moveUp = False
bool moveDown = False
long moveSpeed = False
long yval = 0
long ymax =20 //some max height for the setup
long ymin = 1 // This would be some minimum y value

bool 
void setup() {
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
  
  Serial.begin(9600); 
}

void loop() {
  // Get the position of the stage with the ultrasonic sensor
  digitalWrite(trigPin,LOW)
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance= duration*0.34/2; // in mm

  
  
  // Prints the distance on the Serial Monitor
  Serial.print("Distance: ");
  Serial.println(distance);

  /*
  if y<ymax:
    moveUp = True;
  else:
    moveUp = False;
  if y>ymin
    moveDown = True;
  else:
    moveDown = False;
  */

  

}
