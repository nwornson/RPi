from picamera import PiCamera



h = 1024
cam_res = (int(h),int(.75*h))


cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))

cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])
data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8) 



cam.capture(data,'rgb')