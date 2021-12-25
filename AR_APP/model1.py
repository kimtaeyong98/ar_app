from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3c
from direct.actor.Actor import Actor
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
import direct.gui.DirectGui as dui
from panda3d.core import DirectionalLight
from panda3d.core import LMatrix4f
from panda3d.core import LineSegs
from panda3d.core import NodePath
from panda3d.core import TransparencyAttrib

import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import math

fr = cv.FileStorage("./camera_parameters.txt", cv.FileStorage_READ)
if not fr.isOpened():
    raise IOError("Cannot open cam parameters")
intrisic_mtx = fr.getNode('camera intrinsic matrix').mat()
dist_coefs = fr.getNode('distortion coefficient').mat()
newcameramtx = fr.getNode('camera new intrinsic matrix').mat()
print("camera intrinsic matrix:\n", intrisic_mtx)
print("distortion coefficients: ", dist_coefs.ravel())
print("camera new intrinsic matrix:\n", newcameramtx)
fr.release()

vid_cap = cv.VideoCapture(0, cv.CAP_DSHOW) 
# Check if the webcam is opened correctly
if not vid_cap.isOpened():
    raise IOError("Cannot open webcam")

vid_cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
vid_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
frame_w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(frame_w, frame_h)

base = ShowBase()
base.disableMouse()
winProp = p3c.WindowProperties()
winProp.setSize(frame_w, frame_h)
base.win.requestProperties(winProp)


# 액터1
pandaActor = Actor("Dinosaur.glb",
                   {"walk":"Dinosaur.glb"})
# 액터1 애니메이션
pandaActor.loop("walk")

#액터1
lMinPt, lMaxPt = p3c.Point3(), p3c.Point3()
pandaActor.calcTightBounds(lMinPt, lMaxPt) # in the object space
print(lMinPt, lMaxPt)
pandaActor.setScale(30, 30, 30)
pandaActor.setR(180)
pandaActor.reparentTo(base.render)

#랜더링 효과

pandaActor.setTransparency(TransparencyAttrib.MAlpha)
pandaActor.setAlphaScale(0.5)


#라인
lines = LineSegs()
lines.setColor(255,0,0)
lines.moveTo(0,0,0)
lines.drawTo(3,0,0)
lines.setColor(0,255,0)
lines.moveTo(0,0,0)
lines.drawTo(0,3,0)
lines.setColor(0,0,255)
lines.moveTo(0,0,0)
lines.drawTo(0,0,3)
lines.setThickness(10)
node = lines.create()
n = NodePath(node)
n.reparentTo(pandaActor)
n.setAlphaScale(2)

# set up a texture for (h by w) rgb image♥
tex = p3c.Texture()
tex.setup2dTexture(frame_w, frame_h, p3c.Texture.T_unsigned_byte, p3c.Texture.F_rgb)

#tex.setRamImage(bg_images[0])
background = OnscreenImage(image=tex) # Load an image object
background.reparentTo(base.render2dp)
# We use a special trick of Panda3D: by default we have two 2D renderers: render2d and render2dp, the two being equivalent. We can then use render2d for front rendering (like modelName), and render2dp for background rendering.
base.cam2dp.node().getDisplayRegion(0).setSort(-20) # Force the rendering to render the background image first (so that it will be put to the bottom of the scene since other models will be necessarily drawn on top)

# tracking the aruco marker
pattern_size = (2, 2)
pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
square_size =40.0 # mm unit
pattern_points *= square_size

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)
arucoParams = cv.aruco.DetectorParameters_create()
matView=[]


def updateBg(task):
    def draw(img, axis_center, imgpts):
                    cv.line(img, axis_center, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5) # X
                    cv.line(img, axis_center, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5) # Y
                    cv.line(img, axis_center, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5) # Z
        
    def getViewMatrix(rvecs, tvecs):
                    # build view matrix
                    rmtx = cv.Rodrigues(rvecs)[0]
                    view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                            [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                            [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                            [0.0       ,0.0       ,0.0       ,1.0    ]])
                    inverse_matrix = np.array([[ 1.0, 1.0, 1.0, 1.0],
                                            [-1.0,-1.0,-1.0,-1.0],
                                            [-1.0,-1.0,-1.0,-1.0],#왼손좌표계 -> 오른손좌표계
                                            [ 1.0, 1.0, 1.0, 1.0]])
                    view_matrix = view_matrix * inverse_matrix
                    view_matrix = np.transpose(view_matrix)
                    return p3c.LMatrix4(view_matrix[0][0],view_matrix[0][1],view_matrix[0][2],view_matrix[0][3],
                            view_matrix[1][0],view_matrix[1][1],view_matrix[1][2],view_matrix[1][3],
                            view_matrix[2][0],view_matrix[2][1],view_matrix[2][2],view_matrix[2][3],
                            view_matrix[3][0],view_matrix[3][1],view_matrix[3][2],view_matrix[3][3])
                     
    success, frame = vid_cap.read()
    
    frame= cv.undistort(frame, intrisic_mtx,dist_coefs,None,newcameramtx)
    if success:
        pandaActor.hide()
        textObject.setText("No AR Marker Detected")
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            textObject.setText("[INFO] ArUco marker ID: {}".format(ids))
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID==1:
                    pandaActor.show()
                    corners = markerCorner.reshape((4, 2))

                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                                                    intrisic_mtx, None)

                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrisic_mtx, None)
                    
                    draw(frame, tuple(topLeft.ravel().astype(int)), imgpts)
                    
                    matView = getViewMatrix(rvecs, tvecs)        
                    matViewInv = matView
                    matViewInv.invertInPlace()

                    cam_pos = matViewInv.xformPoint(p3c.LPoint3(0, 0, 0))
                    cam_view = matViewInv.xformVec(p3c.LVector3(0, 0, -1))
                    cam_up = matViewInv.xformVec(p3c.LVector3(0, 1, 0))
                    

                    # camera
                    base.camera.setPos(cam_pos)
                    base.camera.lookAt(cam_pos + cam_view, cam_up)

                    fov_x = 2 * math.atan(frame_w/(2 * intrisic_mtx[0][0])) * 180 / math.pi
                    fov_y = 2 * math.atan(frame_h/(2 * intrisic_mtx[1][1])) * 180 / math.pi
                    base.camLens.setNearFar(10, 10000)
                    base.camLens.setFov(fov_x, fov_y)
                    
                    tmodela = p3c.LMatrix4.translateMat(square_size/2,square_size/2,0) 
                    pandaActor.setMat(tmodela) 
                    pandaActor.setScale(10, 10, 10)
                    pandaActor.setR(180) 
                    
        frame = cv.flip(frame, 0)
        # overwriting the memory with new frame
        tex.setRamImage(frame)
        return task.cont
        

base.taskMgr.add(updateBg, 'video frame update')

aspect = frame_w / frame_h
textObject = OnscreenText(text="No AR Marker Detected", pos=(aspect - 0.05, -0.95), 
                        scale=(0.07, 0.07),
                        fg=(1, 0.5, 0.5, 1), 
                        align=p3c.TextNode.A_right,
                        mayChange=1)
textObject.reparentTo(base.aspect2d)

base.run()

