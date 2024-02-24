#include <ESP8266WiFi.h>
#include "AdafruitIO_WiFi.h"

#define WIFI_SSID       "YOUR SSID"
#define WIFI_PASS       "YOUR WIFI PASSWORD"
#define IO_USERNAME    "ADAFRUI IO USERNAME"
#define IO_KEY         "ADAFRUIT IO KEY"

//Switches
#define R1S1_relay D5
#define R1S2_relay D6

// Connect to Wi-Fi and Adafruit IO handel
AdafruitIO_WiFi io(IO_USERNAME, IO_KEY, WIFI_SSID, WIFI_PASS);

// Create a feed object that allows us to send data to
AdafruitIO_Feed *R1S1 = io.feed("r1s1");
AdafruitIO_Feed *R1S2  = io.feed("r1s2");

void setup()
{
  //Set pin mode
  pinMode(R1S1_relay, OUTPUT);
  pinMode(R1S2_relay, OUTPUT);

  // Enable the serial port so we can see updates
  Serial.begin(115200);
  // Connect to Adafruit IO
  io.connect();
  // wait for a connection
  while (io.status() < AIO_CONNECTED){
    Serial.print(".");
    delay(500);
  }
  //On message call function
  R1S1->onMessage(R1S1_handle);
  R1S2->onMessage(R1S2_handle);
  //Read the last status
  R1S1->get();
  R1S2->get();
}

void loop(){
  io.run();

}
//For room1 switch 1
//If received 1 from feed make pin LOW(switch on)
//If received 0 from feed make pin HIGH(switch on)
void R1S1_handle(AdafruitIO_Data *data) {
  Serial.print("received <- ");
  if (data->toPinLevel() == HIGH)
    Serial.println("HIGH");
  else
    Serial.println("LOW");

  digitalWrite(R1S1_relay, !(data->toPinLevel()));
}
//For room1 switch 2
//If received 1 from feed make pin LOW(switch on)
//If received 0 from feed make pin HIGH(switch on)
void R1S2_handle(AdafruitIO_Data *data) {
  Serial.print("received <- ");
  if (data->toPinLevel() == HIGH)
    Serial.println("HIGH");
  else
    Serial.println("LOW");

  digitalWrite(R1S2_relay, !(data->toPinLevel()));
}
