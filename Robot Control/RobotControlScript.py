#!/usr/bin/env python
from niryo_one_python_api.niryo_one_api import *
import rospy
import time
import math
from std_msgs.msg import String

dataMem = ""

def callback(recieved):
    global n
    global dataMem
    global zPos
    global xPos
    global yPos
    global offset
    print("Recieved Message: " + str(recieved.data))

    #Pulls data out from between square brackets: [x_coord,y_coord,class]
    message = str(recieved.data)
    dataSplit = str(message).split('[')
    dataSplit = str(dataSplit[1]).split(']')
    data = dataSplit[0]

    #Checks if the data is different to previous callback
    if(data != dataMem):
        dataMem = data

        dataElements = str(data).split(',')

        visionX = float(dataElements[0])
        visionY = float(dataElements[1])
        objClass = float(dataElements[2])
    
        if(visionX == 1000):
            return

        xRatio = visionX/xDim
        yRatio = visionY/yDim

        rBoxWidth = abs(yMarks[0]) + abs(yMarks[1])
        rBoxHeight = abs(xMarks[0]) - abs(xMarks[2])

        boxX = (yRatio * rBoxHeight)
        robotX = xMarks[3] + boxX
        xPos = robotX

        boxY = (1-xRatio) * rBoxWidth
        robotY = yMarks[0] - boxY


#       beltOffset = (endOffset - startOffset)*(xRatio)+startOffset

#       yPos = yPos + beltOffset
        print("Xpos = " + str(xPos) + ", yPos = " + str(yPos))

        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

        #Pick and Place Movement
        zPos = pickHeight
        xPos = xPos - offset
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        #n.pull_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)
        n.pull_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)

        #Rise and Rotate
        zPos = zPosDef
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

        #Place wood to the right, place everything else to the left
        if((objClass == 0) or (objClass == 1) or (objClass == 2) or (objClass == 3)):
            yPos = yPosDropLeft
        elif(objClass == 4):
            yPos = yPosDropRight

        xPos = xPosDrop
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

        #Drop
        zPos = 0.1
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        n.push_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)
        n.push_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)

        #Return to default height
        zPos = zPosDef

        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
    else:
        return

rospy.init_node('niryo_one_example_python_api')

#Default Coordinates
xPosDef = 0.2
yPosDef = 0
zPosDef = 0.2

#Drop Coordinates
yPosDropRight = -0.3
yPosDropLeft = 0.3
xPosDrop = 0.05

#Set Coordinates to Default
xPos = xPosDef
yPos = yPosDef
zPos = zPosDef

#CalibrationMarkersCoordinates
xMarks = [0]*4
yMarks = [0]*4

xMarks[0] = 0.35
xMarks[1] = 0.35
xMarks[2] = 0.15
xMarks[3] = 0.15

yMarks[0] = 0.2
yMarks[1] = -0.2
yMarks[2] = -0.2
yMarks[3] = 0.2

#Vision Dimensions
xDim=415
yDim=311

#Misc Arm Parameters
pickHeight = 0.04
increment = 0.05
offset = 0

startOffset = 0.12
endOffset = 0.22

#Program variables
notCalibrated = True
takingInput = True

#Setup
n = NiryoOne()
n.change_tool(TOOL_VACUUM_PUMP_1_ID)
n.calibrate_auto()
print("Calibration finished...")
time.sleep(1)
n.set_arm_max_velocity(100)

print("Application Started...")
print("Moving arm to default position: " + str(xPos) + ", " + str(yPos) + ", " + str(zPos))

n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

while(True):
    print("Press q to trace predefined coordinates and begin input")
    print("Press i to go straight to input coordinates")
    print("Press m to set new coordinates")

    option = raw_input()
    if((option == "i") or (option == "q")):
        if(option == "q"):
            i = 0
            while(i < 4):
                xPos = xMarks[i]
                yPos = yMarks[i]
                enter = raw_input("Press Enter to move to next point...")
                i += 1
                n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        messageMem = ""

        #Read Topic, program stays at .spin() until end
        rospy.Subscriber('coordinates', String, callback, queue_size = 1)
        rospy.spin()

    elif(option == "m"):
        print("Use WASD to move the arm and lock in the coordinates with m...")
        markerNum = 0
        while(notCalibrated):
            keyboardPress = raw_input()
            if(keyboardPress == 'w'):
                print("Increasing x value by " + str(increment) + "...")
                xPos = xPos + increment
            elif(keyboardPress == 's'):
                print("Decreasing x value by " + str(increment) + "...")
                xPos = xPos - increment
            elif(keyboardPress == 'a'):
                print("Increasing y value by " + str(increment) + "...")
                yPos = yPos + increment
            elif(keyboardPress == 'd'):
                print("Decreasing y value by " + str(increment) + "...")
                yPos = yPos - increment
            elif(keyboardPress == 'm'):
                xMarks[markerNum] = xPos
                yMarks[markerNum] = yPos
                markerNum += 1
                print("Added marker number " + str(markerNum))
                if(markerNum > 3):
                    notCalibrated = False
            n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
            print("Current Coordinates, X: " + str(xPos) + ", Y: " + str(yPos))

n.activate_learning_mode(True)
print("Control Application Finished...")

#scp RobotControlScript.py niryo@192.168.1.102:catkin_ws/src/niryo_one_python_api/scripts/







