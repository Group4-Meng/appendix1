#!/usr/bin/env python

import rospy

from std_msgs.msg import String 

import numpy as np
import time

print('1')

def callback(recieved):
        print('Data Recieved',recieved.data)
        #rospy.loginfo(rospy.get_caller_id()+'message is %s ',recieved.data)


rospy.init_node('listener', anonymous=True)

print('2')

 
rospy.Subscriber('chatter',String, callback)

print('3')

i=0

initial_time=time.time()

while not rospy.core.is_shutdown():
	
	
	if time.time()-initial_time>10:
		break

	
	
		
	rospy.rostime.wallsleep(0.5)

print('OUTTTTTTTTTTTTTTTTTTTTTT')
	


