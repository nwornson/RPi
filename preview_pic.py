'''
https://makersportal.com/blog/2019/4/21/image-processing-using-raspberry-pi-and-python

'''

from time import sleep
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (2560,1936)
# rotate
camera.rotation = 180
#camera.start_preview()
sleep(5)
camera.capture('test.jpg')
camera.stop_preview()