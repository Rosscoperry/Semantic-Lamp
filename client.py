from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect
from functions import semanticlamp
import logging


class Client(object):
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        PeriodicCallback(self.keep_alive, 10000).start()
        self.ioloop.start()

    @gen.coroutine
    def connect(self):
        """
        ***write docstring***
        """
        print("trying to connect")
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception:
            print("connection error")
        else:
            print("connected")
            self.run()

    @gen.coroutine
    def run(self):
        """
        ***write docstring***
        """

        while True:
            print('What do you have to say?')
            captured_text = semanticlamp.capture()  # get microphone text
            print(captured_text)

            if captured_text:
                if "quit" in str(captured_text):
                    print("okay, bye.")
                    print("connection closed")
                    self.ws = None
                    exit()
                else:
                    print(f"Heard: {captured_text}")
                    # predict sentiment of captured text
                    label, score = semanticlamp.predict(captured_text)
                    print(f"Predicted sentiment: {label}, Score: {score} ")
                    # calculate message to send to server
                    msg = semanticlamp.update_colour(label, score)
                    print(msg)

                    if self.ws is None:
                        self.connect()
                    else:
                        self.ws.write_message(str(msg))

            # mic live

            # msg = yield predict_sentiment()
                # upon running program mic must be live
                # when using command "quit" the mic must close and the connections must close
                # program must send an array indicating a hex colour.
                # server must interpret and have two bars labeled G ########## and B ########## levels adjust according to sentiment

    def keep_alive(self):
        """
        ***write docstring***
        """
        if self.ws is None:
            self.connect()
        else:
            self.ws.write_message("hello")


# load model
semanticlamp = semanticlamp.SemanticLamp()

if __name__ == "__main__":
    client = Client("ws://xxx.xxx.x.xx:8888", 5)


# * * * * * python /root/python/tempLCD.py
# #
# */1 * * * * /root/rgb-led.sh
# # runs led on extension board
