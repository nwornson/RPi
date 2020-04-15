# statistically detect differences between frames




from picamera import PiCamera
from scipy import stats
import imageio
import time
import numpy as np
# custom ttest function to obtain pvalue
def ttest_one_sample(data):
    N = len(data)
    xbar = sum(data)/N
    var = (sum((data-xbar)**2)/(N - 1))
    cv = xbar / ((var/N)**.5)
    pval = 1 - stats.t.cdf(abs(cv),df=N-1)
    return(pval)

# camera resolution
h = 512 #1024
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
ite = 0
while True:
    try:
        # capture the image and store it
        cam.capture(data_cur,'rgb')

        # compare to previous image and apply t-test
        
        pmu = data_prev.mean()
        cmu = data_prev.mean()
        pstd = (data_prev-pmu)**2/(len(data_prev)-1)
        cstd = (data_cur-pmu)**2/(len(data_cur)-1)
        
        norm_prev = (data_prev - pmu)/pstd
        norm_cur = (data_cur - cmu)/cstd
        
        diff = norm_cur - norm_prev
        diff = diff.flatten()
        pvalue = ttest_one_sample(diff)

        # if the image is significantly different than the last, save it
        if pvalue < .001:
            # save to file named as current time
            file_name = time.asctime( time.localtime(time.time()) )
            file_name = str.split(file_name)[3].replace(':',"")
            img = np.uint8(data_cur)
            imageio.imwrite(file_name + '.png', img)

        data_prev = data_cur
        data_cur = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8)
        
        ite +=1
        
        print([ite,pvalue,type(pvalue)])
        

    except KeyboardInterrupt:
        break
