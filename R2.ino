#include <ESP8266WiFi.h>
#include "AdafruitIO_WiFi.h"

#define WIFI_SSID       "YOUR SSID"
#define WIFI_PASS       "YOUR WIFI PASSWORD"
#define IO_USERNAME    "ADAFRUI IO USERNAME"
#define IO_KEY         "ADAFRUIT IO KEY"

//Switches
#define R2S1_relay D5
#define R2S2_relay D6

// Connect to Wi-Fi and Adafruit IO handel
AdafruitIO_WiFi io(IO_USERNAME, IO_KEY, WIFI_SSID, WIFI_PASS);

// Create a feed object that allows us to send data to
AdafruitIO_Feed *R2S1 = io.feed("r2s1");
AdafruitIO_Feed *R2S2  = io.feed("r2s2");

void setup()
{
  pinMode(R2S1_relay, OUTPUT);
  pinMode(R2S2_relay, OUTPUT);
  
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
  R2S1->onMessage(R2S1_handle);
  R2S2->onMessage(R2S2_handle);
  //Read the last status
  R2S1->get();
  R2S2->get();
}

void loop(){
  io.run();
  
}
//For room2 switch 2
//If received 1 from feed make pin LOW(switch on)
//If received 0 from feed make pin HIGH(switch on)
void R2S1_handle(AdafruitIO_Data *data){
  Serial.print("received <- ");
  if(data->toPinLevel() == HIGH)
    Serial.println("HIGH");
  else
    Serial.println("LOW");
    
  digitalWrite(R2S1_relay, !(data->toPinLevel()));
}

//For room2 switch 2
//If received 1 from feed make pin LOW(switch on)
//If received 0 from feed make pin HIGH(switch on)
void R2S2_handle(AdafruitIO_Data *data){
  Serial.print("received <- ");
  if(data->toPinLevel() == HIGH)
    Serial.println("HIGH");
  else
    Serial.println("LOW");
    
  digitalWrite(R2S2_relay, !(data->toPinLevel()));
}
