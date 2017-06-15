#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:18:57 2017
Tornado
@author: wangx3
"""

'''
import socket
 
def handle_request(client):
    # 获取请求头信息
    buf = client.recv(1024)
    # 5.2 回传响应
    client.send("HTTP/1.1 200 OK\r\n\r\n")
    client.send("Hello, Seven")
 
def main():
    # 1. 建立一个Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2. 绑定端口
    sock.bind(('localhost',8080))
    # 3. 监听端口
    sock.listen(5)
 
    while True:
        # 4. 建立连接
        connection, address = sock.accept()
        # 5. 处理请求
        handle_request(connection)
        # 6. 断开连接
        connection.close()
 
if __name__ == '__main__':
    main()
'''
 

import os.path

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient
import tornado.gen

import urllib
import json
import datetime
import time

'''
# 定义处理类的回传响应 
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Wang Xin")

# 创建一个application对象，生成url映射
# 把url正则表达（/index）式和处理类（MainHandler）配对
application = tornado.web.Application([
    (r"/index", MainHandler),
])
 
if __name__ == "__main__":
    # 指定监听端口
    application.listen(8888)
    # 执行监听和响应
    tornado.ioloop.IOLoop.instance().start()
'''
'''   
from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        # 定义URL映射
        handlers = [
            (r"/", MainHandler),
        ]
        # 定义动态和静态路径
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            debug=True,
            xsrf_cookies=True,
            gzip=False
        )
        # 用handlers和settings定义Application类的生成函数
        tornado.web.Application.__init__(self, handlers, **settings)

# 定义继承RequestHandler类的应用子类        
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # 渲染选定的htmlwenjian
        self.render(
            "index.html",
            page_title = "Burt's Books | Home",
            header_text = "Welcome to Burt's Books!",
        )

if __name__ == "__main__":
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
'''


from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous     #声明异步装饰器，Tornado返回请求后不关闭连接
    def get(self):
        query = self.get_argument('q')
        client = tornado.httpclient.AsyncHTTPClient()
        client.fetch("http://search.twitter.com/search.json?" + \
                urllib.urlencode({"q": query, "result_type": "recent", "rpp": 100}),
                callback=self.on_response)    #指定回调函数

    def on_response(self, response):
        # 回调函数通过json格式获取服务器返回内容
        body = json.loads(response.body)
        result_count = len(body['results'])
        now = datetime.datetime.utcnow()
        raw_oldest_tweet_at = body['results'][-1]['created_at']
        oldest_tweet_at = datetime.datetime.strptime(raw_oldest_tweet_at,
                "%a, %d %b %Y %H:%M:%S +0000")
        seconds_diff = time.mktime(now.timetuple()) - \
                time.mktime(oldest_tweet_at.timetuple())
        tweets_per_second = float(result_count) / seconds_diff
        self.write("""
<div style="text-align: center">
    <div style="font-size: 72px">%s</div>
    <div style="font-size: 144px">%.02f</div>
    <div style="font-size: 24px">tweets per second</div>
</div>""" % (self.get_argument('q'), tweets_per_second))
        self.finish()    # 在回调函数中明示关闭连接

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
    