from picamera import PiCamera
import imageio
import time
import numpy as np

def capture(h):

# camera resolution
    cam_res = (int(h),int(.75*h))
    cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))

    # initialize camera
    cam = PiCamera()
    cam.resolution = (cam_res[1],cam_res[0])

    # initialize data
    data_prev = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8) 
    data_cur = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8) 

    # initial image capture
    cam.capture(data_prev,'rgb')

    file_name = time.asctime( time.localtime(time.time()) )
    file_name = str.split(file_name)[3].replace(':',"")
    img = np.uint8(data_cur)
    imageio.imwrite(file_name + '.png', img)

    return img


