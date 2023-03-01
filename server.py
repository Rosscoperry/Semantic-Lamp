import logging
import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.options
import os

from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)
clients = []


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/", MainHandler)]
        settings = dict(debug=True)
        tornado.web.Application.__init__(self, handlers, **settings)


class MainHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        logging.info("A client connected.")
        clients.append(self)

    def on_close(self):
        logging.info("A client disconnected")
        clients.remove(self)
        os.system("expled 0x000000")

    def on_message(self, message):

        logging.info(message.decode("utf-8"))
        os.system("expled " + message)


def main():
    tornado.options.parse_command_line()
    app = Application()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
