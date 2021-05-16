import gxipy as gx
from PIL import Image	
import cv2
import imutils
import numpy as np
    #Device Manager for running the camera
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num is 0:
	print("Number of enumerated devices is 0")

	#Open the camera and create a camera object
cam = device_manager.open_device_by_index(1)

	###########################################CAMERA PARAMETERS##########################################

	# set continuous acquisition
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

	# set exposure
cam.ExposureTime.set(30000.0)

cam.BalanceWhiteAuto.set(True)

	# set gain
cam.Gain.set(10.0)

	# get param of improving image quality
if cam.GammaParam.is_readable():
	gamma_value = cam.GammaParam.get()
	gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
else:
	gamma_lut = None
if cam.ContrastParam.is_readable():
	contrast_value = cam.ContrastParam.get()
	contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
else:
	contrast_lut = None
if cam.ColorCorrectionParam.is_readable():
	color_correction_param = cam.ColorCorrectionParam.get()
else:
	color_correction_param = 0

cam.Width.set(1080)
cam.Height.set(1080)

	###########################################CAMERA PARAMETERS END#######################################

	# start data acquisition
cam.stream_on()

picturenum = 300
while(True):
    raw_image = cam.data_stream[0].get_image()
    rgb_image = raw_image.convert("RGB")
    numpy_image = rgb_image.get_numpy_array()
    im = Image.fromarray(numpy_image)
    im_rgb = imutils.resize(numpy_image.astype(np.uint8))
    im_rgb_stream = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
    # show acquired image
    #img = Image.fromarray(numpy_image, 'L')
    cv2.imshow("Stream", im_rgb_stream)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        #print("Taking Image...")
        im = Image.fromarray(im_rgb)
        im.save("pictures/" + str(picturenum) + ".jpeg")
        print("Image number " + str(picturenum) + " saved!")
        picturenum = picturenum + 1

    

    
