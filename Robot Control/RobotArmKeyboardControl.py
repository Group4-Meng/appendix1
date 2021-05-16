import keyboard
from niryo_one_python_api.niryo_one_api import *
import rospy
import time
import math

#Default Coordinates
xPosDef = 0.2
yPosDef = 0
zPosDef = 0.2

#Drop Coordinates
xPosDrop = 0
yPosDrop = -0.3

#Set Coordinates to Default
xPos = xPosDef
yPos = yPosDef
zPos = zPosDef

#CalibrationMarkersCoordinates
xMarks = [0]*4
yMarks = [0]*4

#Vision Dimensions
xDim=416
yDim=416

print("Control Application Start...")

running = True
markersCalibrated = False

#Setup
n = NiryoOne()
n.change_tool(TOOL_VACUUM_PUMP_1_ID)
n.calibrate_auto()
print("Calibration finished...")
time.sleep(1)
n.set_arm_max_velocity(30)

print("Use WASD to move the arm")
print("W--S: X axis")
print("A--D: Y axis")
print("Enter initiates a pick movement")
while(running):
    #Move to default position
    n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
    
    #Control Loop
    keyboardDetect()

    if((markersCalibrated == True) and (running == True)):
        visionX = float(input("Input X value from vision system: "))
        visionY = float(input("Input Y value from vision system: "))

        xPos = (visionX/xDim)*(xMarks[0]-xMarks[1])
        yPos = (visionY/yDim)*(yMarks[0]-yMarks[2])

        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        input1 = input("Press Enter to pick up object...")

    if(running == True):
        #Pick and Place Movement
        #Pick
        zPos = 0.04
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        n.pull_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)
        n.pull_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)

        #Rise and Rotate
        zPos = zPosDef
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        xPos = xPosDrop
        yPos = yPosDrop
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

        #Drop
        zPos = 0.1
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)
        n.push_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)
        n.push_air_vacuum_pump(TOOL_VACUUM_PUMP_1_ID)

        #Return to default
        zPos = zPosDef
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

        xPos = xPosDef
        yPos = yPosDef

n.activate_learning_mode(True)
print("Control Application Finished...")

def keyboardDetect():
    global xPos
    global yPos
    global zPos
    global runnning 
    global n
    global xMarks
    global yMarks
    global markersCalibrated
    markerNum = 0
    while(True):
        if(keyboard.is_pressed('w')):
            print("Increasing x value by 0.01...")
            xPos = xPos + 0.01
        elif(keyboard.is_pressed('s')):
            print("Decreasing x value by 0.01...")
            xPos = xPos - 0.01
        elif(keyboard.is_pressed('a')):
            print("Increasing y value by 0.01...")
            yPos = yPos + 0.01
        elif(keyboard.is_pressed('d')):
            print("Decreasing y value by 0.01...")
            yPos = yPos - 0.01
        elif(keyboard.is_pressed('m')):
            xMarks[markerNum] = xPos
            yMarks[markerNum] = yPos
            markerNum += 1
            print("Added marker number " + str(markerNum))
            if(markerNum > 3):
                markersCalibrated = True
                return
        elif(keyboard.is_pressed('r')):
            print("Removing all calibration point entries...")
            markerNum = 0
            for x, y in zip(xMarks, yMarks):
                x = 0
                y = 0
        elif(keyboard.is_pressed('enter')):
            print("Initiating Pick...")
            return
        elif(keyboard.is_pressed('esc')):
            print("Control Exited...")
            running = False
            return
        n.move_pose(xPos, yPos, zPos, 0, math.radians(90), 0)

#scp RobotArmKeyboardControl.py pi@192.168.1.102:folder1/