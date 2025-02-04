#include <Servo.h>

Servo myservo;  // create servo object to control a servo

void setup() {
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600); // initialize serial communication
  Serial.println("Setup complete");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    Serial.print("Received command: ");
    Serial.println(input);

    int position = input.toInt(); // extract position from the command
    Serial.print("Parsed position: ");
    Serial.println(position);

    if (position >= 0 && position <= 180) {
      myservo.write(position);  // move the servo to the desired position
      Serial.print("Moving to position: ");
      Serial.println(position); // confirm the position
      Serial.println("Move complete");
    } else {
      Serial.println("Invalid position");
    }
  }
}
