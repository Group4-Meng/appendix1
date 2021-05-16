#!/usr/bin/env python3
#import sys
#ROS_PATH_1='/home/tarek/catkin_ws/devel/lib/python2.7/dist-packages'
#ROS_PATH_2='/opt/ros/kinetic/lib/python2.7/dist-packages'
#Ros_path_list=[ROS_PATH_1,ROS_PATH_2]
#print('Initial sys path',sys.path)

#for path in range(1,3):
#	sys.path.remove(sys.path[1]) 

import cv2


#for i in Ros_path_list:
#	print('Appending',i)
#	sys.path.append(i)

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Float32
from std_msgs.msg import Int32


import argparse
import time
from pathlib import Path
import gxipy as gx
import cv2
import torch
import torch.backends.cudnn as cudnn
import imutils
import numpy as np
from numpy import random
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
img_size = 416

rospy.init_node('YOLO_V5_pub',anonymous=True)
publisher=rospy.Publisher('coordinates',String, queue_size=1000)
msg=String()
#rate=rospy.Rate(100)




Tensor = torch.cuda.HalfTensor
posList = []
M = []
height = 0
width = 0
def cameraSetup():
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
	cam.ExposureTime.set(4000.0)
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
	cam.Width.set(1440)
	cam.Height.set(1080)
	###########################################CAMERA PARAMETERS END#######################################
	
	#Start data acquisition
	cam.stream_on()
	return

def cameraAcquisition():
	raw_image = cam.data_stream[0].get_image()

	########Conversion to CV format nd correct colour#############
	rgb_image = raw_image.convert("RGB")
	numpy_image = rgb_image.get_numpy_array()
	im_rgb = imutils.resize(numpy_image.astype(np.uint8), width = 415)
	#im_rgb = im_rgb[0:192, 0:416]
	########Conversion to CV format and colour end################
	return im_rgb


def onMouse(event, x, y, flags, param):
	global posList
	if (event == cv2.EVENT_LBUTTONDOWN) and (len(posList) < 4):
		posList.append((x, y))

def getWarp():
	global M
	global width
	global height
	firstException = True
	while(True):
		im_rgb = cameraAcquisition()
		im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
		for points in posList:
			cv2.circle(im_rgb, points, 5, (0, 0, 255), -1)
		try:	
			cv2.line(im_rgb,posList[0],posList[1],(255,0,0),2)
			cv2.line(im_rgb,posList[1],posList[3],(0,255,0),2)
			cv2.line(im_rgb,posList[3],posList[2],(0,0,255),2)
			cv2.line(im_rgb,posList[2],posList[0],(255,0,255),2)
			height, width, channels = im_rgb.shape
						
			srcPoints = np.float32([posList[0], posList[1], posList[3], posList[2]])
			destPoints = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])

			M = cv2.getPerspectiveTransform(srcPoints, destPoints)
			warped = cv2.warpPerspective(im_rgb, M, (width, height))
			print("Height of warped image: " + str(height) + ", Width of warped image: " + str(width))
			cv2.imshow("warped", warped)
		except:
			if(firstException):
				print("Select 4 Points...")
				firstException = False

		cv2.imshow("Markers", im_rgb)
		cv2.setMouseCallback('Markers', onMouse)

		key = cv2.waitKey(1) & 0xFF
			#Press q to lock in current warp
		if key == ord("q"):
			notCalibrated = False
			cv2.destroyAllWindows()
			return M
		if key == ord("d"):
			firstException = True
			cv2.destroyAllWindows()
			posList.clear()





def detect(m):
	global img_size
	global Tensor
	global height
	global width
	global NO_DETECTION
	source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://'))
	#webcam = cam.data_stream[0]


	# Directories
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Initialize
	set_logging()
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	#imgsz = check_img_size(imgsz, s=stride)  # check img_size
	imgsz = img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	#if webcam:
	view_img = check_imshow()
	cudnn.benchmark = True  # set True to speed up constant image size inference
	#    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
	#else:
	#    dataset = LoadImages(source, img_size=imgsz, stride=stride)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
	print('NAMES_LIST',names)

	# Run inference
	if device.type != 'cpu':
		print("Device = CUDA")
		model(torch.zeros(1, 3, 416, 416).to(device).type_as(next(model.parameters())))  # run once
	t0 = time.time()
	vid = cv2.VideoCapture(0)
	frameNum = -1
	#Loop for image acquisition
	while(True): #path, img, im0s, vid_cap in dataset:
		#print("Loop Start")
		frame = cameraAcquisition()
		warped = cv2.warpPerspective(frame, m, (width, height))
		frame = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
		pilimg = Image.fromarray(frame)



		ratio = min(img_size/pilimg.size[0], img_size/pilimg.size[1])
		imw = round(pilimg.size[0] * ratio)
		imh = round(pilimg.size[1] * ratio)
		img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
		 transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
						(128,128,128)),
		 transforms.ToTensor(),
		 ])


		image_tensor = img_transforms(pilimg).float()
		image_tensor = image_tensor.unsqueeze_(0)
		input_img = Variable(image_tensor.type(Tensor))
		
		
		tran1 = transforms.ToPILImage()
		pil_image_single = tran1(input_img[0])
		open_cv_image = np.array(pil_image_single)
		cv2.imshow("Image", open_cv_image)
		
		t1 = time_synchronized()

		pred = model(input_img, augment=opt.augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)
			print('pred is:',pred.shape)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			#p, s, im0, frame = path[i], '%g: ' % i, frame.copy(), frameNum
			print('det is:',det.shape)
			print('det_matrix:',det)
			

			centre_list=[]
			class_list=[]
			s = '%g: ' % i
			#p = Path(p)  # to Path
			#save_path = str(save_dir / p.name)  # img.jpg
			#txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			#s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				#print("Detection")
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(input_img.shape[2:], det[:, :4], frame.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					#s += f'{n} {names[int(c)]}{'s' * (n > 1)}, '  # add to string

				# Write results
				
				for *xyxy, conf, cls in reversed(det):
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or view_img:  # Add bbox to image
						if(cls == 3):
							cls = 2
						elif(cls == 2):
							cls = 3
						

						
						#if(conf > 0.5):
						#	print("Class Number: " + str(cls) + "Class Label: " + str(names[int(cls)]))
						label = f'{names[int(cls)]} {conf:.2f}'
						plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
						c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
						centre=(int((c1[0]+c2[0])/2),int((c1[1]+c2[1])/2))
						centre_list.append(centre)
						class_list.append(cls)

				wood_flag=False
				idx_first=0

				for idx,element in enumerate(class_list):
					if(element==4):
						wood_flag=True
						print('WOOD DETECTED >> Publishing Centre')
						centre_tuple=centre_list[idx]
						break
				

					
				if (wood_flag==False):
					centre_tuple=centre_list[idx_first]	

                  


				print('centre is:',centre_tuple)
				x_cord=centre_tuple[0]
				y_cord=centre_tuple[1]

				if x_cord !=96 or y_cord !=72:

					msg.data= "["+str(x_cord) + "," + str(y_cord) + "," + str(int(cls))+"]"
					publisher.publish(msg)
					print('Published coordinate',centre_tuple)
				else:
					if wood_flag==True:
						print('DETECTION OF CENTER B_BOX as wood>> NO DEC')
					elif wood_flag==False:
						print('DETECTION OF CENTER B_BOX as <<non-wood>> published  NO DEC')
					
					msg.data="["+str(1000)+","+str(1000)+","+str(1000)+"]"
					publisher.publish(msg)
					

					
				#rate.sleep()
	

	
						#plot_one_box(xyxy,centre, im0, label=label, color=colors[int(cls)], line_thickness=10)

					
			else:
				print('No Detections')
				msg.data="["+str(1000)+","+str(1000)+","+str(1000)+"]"
				publisher.publish(msg)
				print('Published>> NO DETECTION')
				
			# Print time (inference + NMS)
			#print(f'Done. ({t2 - t1:.3f}s)')

			# Stream results
			if True:
				cv2.imshow("Yo", frame)
				cv2.waitKey(1)  # 1 millisecond
				

			# Save results (image with detections)
			#if save_img:
			#    if dataset.mode == 'image':
			#        cv2.imwrite(save_path, im0)
			#    else:  # 'video' or 'stream'
			#        if vid_path != save_path:  # new video
			#            vid_path = save_path
			#            if isinstance(vid_writer, cv2.VideoWriter):
			#                vid_writer.release()  # release previous video writer
			#            if vid_cap:  # video
			#                fps = vid_cap.get(cv2.CAP_PROP_FPS)
			#                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			#                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			#            else:  # stream
			#                fps, w, h = 30, im0.shape[1], im0.shape[0]
			#                save_path += '.mp4'
			#            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
			#        vid_writer.write(im0)

	#if save_txt or save_img:
	#    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
	#    print(f"Results saved to {save_dir}{s}")

	#print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	opt = parser.parse_args()
	print(opt)
	check_requirements(exclude=('pycocotools', 'thop'))
	
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
	cam.ExposureTime.set(35000.0)
	# set gain
	cam.Gain.set(10.0)
	cam.BalanceWhiteAuto.set(True)
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
	cam.Width.set(1440)
	cam.Height.set(1080)
	###########################################CAMERA PARAMETERS END#######################################
	
	#Start data acquisition
	cam.stream_on()

	warp = getWarp()

	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
				detect(warp)
				strip_optimizer(opt.weights)
		else:
			detect(warp)
