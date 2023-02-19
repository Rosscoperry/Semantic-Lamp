#import lcdDriver

import tornado.ioloop
import tornado.websocket

#lcdAddress = 0x3f


class IoTWebsocket(tornado.websocket.WebSocketHandler):

    def check_origin(self, origin):
        return True

    def open(self):
        print("Websocket Opened on")

#    def on_message(self, message):
#        displayMsg("Recieved")
    def on_message(self, message):
        print("recieved:", message)

    def on_close(self):
        print("Websocket Closed")


def make_app():
    return tornado.web.Application([
        (r"/", IoTWebsocket),
    ])

# function to display the temperature on the LCD screen
# def displayMsg(msg):
#    # setup LCD
#    lcd = lcdDriver.Lcd(lcdAddress)
#    lcd.backlightOn()
#
#    lcd.lcdDisplayStringList([
#        "Message: ",
#        str(msg) + "."
#    ])


if __name__ == "__main__":

    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
