# Semantic Lamp

A program that performs semantic analysis on the recorded words picked up by a microphone. 

User can choose between running this program entirely locally or integrate into their IoT network. 

## Prerequisites

This program runs on python 3.9.16.

Before starting - create a virtual environment and run 

```
pip install -r requirements.txt
```

You can chose if you want to use the tokenizer and model file provided - these have reasonable accuracy and are a good baseline to get started. 
If you want to have a go at creating these yourself, you can using ```preprocessing.py``` in the ```functions``` folder. 

## Option 1 - Running locally 

navigate to the SEMANTIC-LAMP working directory and run:

```
python semantic-lamp-sim.py
```

This will start the program and use a pygame window to simulate a RGB LED. 

The 'RGB LED' will appear more red if negative utterances are said; more green if positive.

## Option 2 - Running using a Python Socket Clients

Using a Onion Omega 2+ I set up a client/server interaction that enables an RGB LED to be controlled remotely via a wifi connection. 

The only line that need changed is the ip address of the single-board computer (server) as this will be specific to your setup.

On your server, run:
```
python server.py
```
On your client, the device with the microphone, run:
```
python client.py
```

## Thank you, :)

### Connect with me: 

[linkedin]: (https://www.linkedin.com/in/rossalexanderperry/)

#### Fancy trying to improve what I made?
* Write some tests ;) 
* Train a model to detect more than 3 classes (Aggressive, Hopelessness, Love, Calmness)
* Make the recording continuous. 