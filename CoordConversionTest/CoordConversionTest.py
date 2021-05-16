from graphics import *
#pip3 install graphics.py

winRobot = GraphWin('Robot Coordinate',width = 416, height = 311) # create a window
winCamera = GraphWin('Camera Coordinate (Flipped)',width = 416, height = 311) # create a window
winRobot.setCoords(0, 0, 4, 2) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
winCamera.setCoords(416, 0, 0, 311)

startOffset = 0.06
endOffset = 0.2

#Draw Grids
gridlines = 0
while(gridlines < 11):
    ln1 = Line(Point(((416/10)*gridlines), 0), Point(((416/10)*gridlines), 311))
    ln1.draw(winCamera)
    ln1 = Line(Point(((4/10)*gridlines), 0), Point(((4/10)*gridlines), 2))
    ln1.draw(winRobot)
    gridlines = gridlines + 1

gridlines = 0
while(gridlines < 9):
    ln1 = Line(Point(0, ((311/8)*gridlines)), Point(416, ((311/8)*gridlines)))
    ln1.draw(winCamera)
    ln1 = Line(Point(0, ((2/8)*gridlines)), Point(4, ((2/8)*gridlines)))
    ln1.draw(winRobot)
    gridlines = gridlines + 1


yMarks = [0]*4
xMarks = [0]*4

xMarks[3] = 0.1
xMarks[0] = 0.3
yMarks[0] = 0.2
yMarks[1] = -0.2

xDim = 416
yDim = 311

pt = Point(0, 0)
cir = Circle(pt, 3)
cir1 = Circle(pt, 3)

while(True):
    visionX = float(input("Input X value from vision system: "))
    visionY = float(input("Input Y value from vision system: "))

    cir.undraw()
    cir1.undraw()

    pt = Point(visionX, visionY)
    cir = Circle(pt, 3)
    cir.setFill("red")
    cir.draw(winCamera) # draw it to the window

    xRatio = visionX/xDim
    yRatio = visionY/yDim

    #Width and Height of the pick area in niryo coordinates
    rBoxWidth = abs(yMarks[0]) + abs(yMarks[1])
    rBoxHeight = abs(xMarks[0]) - abs(xMarks[3])

    #Coordinate assuming box has no negative part
    boxX = (yRatio * rBoxHeight)
    #Coordinate adjusted to take into account negative and positive parts of box
    robotX = xMarks[3] + boxX
    xPos = robotX

    #Coordinate assuming box has no negative part
    boxY = (1-xRatio) * rBoxWidth
    #Coordinate adjusted to take into account negative and positive parts of box
    robotY = yMarks[0] - boxY
    yPos = robotY

    #offset = (endOffset - startOffset)*(1-xRatio)
    

    #boxY = boxY + offset

    pt = Point(boxY*10, boxX*10)
    cir1 = Circle(pt, 0.03)
    cir1.setFill("red")
    cir1.draw(winRobot) # draw it to the window

    print("Xpos = " + str(xPos) + ", yPos = " + str(yPos))