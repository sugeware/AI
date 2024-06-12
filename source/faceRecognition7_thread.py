from threading import Thread
import time

def BigBox(color):
    while True:
        print(color+' Big box is open')
        time.sleep(1)
        print(color+' Big box is closed')
        time.sleep(1)

def SmallBox(color):
    while True:
        print(color+' Small box is open')
        time.sleep(5)
        print(color+' Small box is closed')
        time.sleep(5)
r = 'red'
bigThread = Thread(target=BigBox,args=(r,))
b = 'blue'
smallThread = Thread(target=SmallBox,args=(b,))

bigThread.daemon = True
smallThread.daemon = True

bigThread.start()
smallThread.start()

while True:
    pass
        
