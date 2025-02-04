#include <Servo.h>

// FSRs connected to analog pins A0, A1, and A2
const int FSR_PIN0 = A0;
const int FSR_PIN1 = A1;
const int FSR_PIN2 = A2;

Servo myservo;  // create servo object to control a servo

void setup() {
  myservo.attach(9);
  Serial.begin(115200);
  delay(1000);  // Wait for the serial port to initialize
  Serial.println("Setup complete");
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    Serial.print("MSG:Received command: ");
    Serial.println(input);

    int position = input.toInt(); // extract position from the command
    Serial.print("MSG:Parsed position: ");
    Serial.println(position);

    if (position >= 0 && position <= 180) {
      myservo.write(position);  // move the servo to the desired position
      Serial.print("MSG:Moving to position: ");
      Serial.println(position); // confirm the position
      Serial.println("MSG:Move complete");
    } else {
      Serial.println("MSG:Invalid position");
    }
  }

  // Read analog values from all three FSRs
  int fsrReading0 = analogRead(FSR_PIN0); // FSR on A0
  int fsrReading1 = analogRead(FSR_PIN1); // FSR on A1
  int fsrReading2 = analogRead(FSR_PIN2); // FSR on A2
//
//  // Convert readings to voltages (assuming 5V Arduino)
//  float voltage0 = fsrReading0 * (5.0 / 1023.0);
//  float voltage1 = fsrReading1 * (5.0 / 1023.0);
//  float voltage2 = fsrReading2 * (5.0 / 1023.0);
//
//  // Convert voltages to forces (adjust calibration as needed)
//  float force0 = voltageToForce(voltage0);
//  float force1 = voltageToForce(voltage1);
//  float force2 = voltageToForce(voltage2);

  // Send force values via serial, separated by commas
  Serial.print("DATA:");
  Serial.print(fsrReading0);
  Serial.print(",");
  Serial.print(fsrReading1);
  Serial.print(",");
  Serial.println(fsrReading2); // End the line after the third value

  delay(10); // Adjust delay as needed
}

float voltageToForce(float voltage) {
  // Convert voltage to force (adjust this based on FSR calibration)
  if (voltage <= 0.5) {
    return 0; // No force
  } else {
    return (voltage - 0.5) * 100; // Example scaling factor
  }
}
