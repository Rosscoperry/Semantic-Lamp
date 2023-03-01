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
        self.websock = None
        self.connect()
        PeriodicCallback(self.keep_alive, 20000).start()
        self.ioloop.start()

    @gen.coroutine
    def connect(self):
        """
        ***write docstring***
        """
        print("trying to connect")
        try:
            self.websock = yield websocket_connect(self.url)
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
                    self.websock = None
                    exit()
                else:
                    print(f"Heard: {captured_text}")
                    # predict sentiment of captured text
                    label, score = semanticlamp.predict(captured_text)
                    print(f"Predicted sentiment: {label}, Score: {score} ")
                    # calculate message to send to server
                    msg = semanticlamp.update_colour(label, score)
                    print(msg)

                    if self.websock is None:
                        self.connect()
                    else:
                        self.websock.write_message(str(msg))

    def keep_alive(self):
        """
        ***write docstring***
        """
        if self.websock is None:
            self.connect()
        else:
            self.websock.write_message("hello")


# load model
semanticlamp = semanticlamp.SemanticLamp()

if __name__ == "__main__":
    client = Client("ws://xxx.xxx.x.xx:8888", 5)
