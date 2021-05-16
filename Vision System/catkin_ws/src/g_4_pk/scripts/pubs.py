#!/usr/bin/env python

import rospy

from std_msgs.msg import String 

import numpy as np


rospy.init_node('talker', anonymous=True)


msg=String()
msg.data='Hello, Publisher Speaking'
 
publisher=rospy.Publisher('chatter',String, queue_size=100)



while not rospy.is_shutdown():
	publisher.publish(msg)
	
