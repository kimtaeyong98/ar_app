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

T1 = np.float32([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])
T2 = np.float32([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])
T3 = np.float32([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])
T1_inv = np.float32([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])

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
pandaActor = Actor("두발팬더.bam",
                {"walk": "models/panda-walk"})
# 액터1 애니메이션
pandaActor.loop("walk")

# 액터2
pandaActor2 = Actor("Dinosaur.glb",
                        {"walk2": "Dinosaur.glb"})
# 액터2애니메이션
pandaActor2.loop("walk2")

# 액터3
pandaActor3 = Actor("Fox.glb",
                        {"walk3": "Fox.glb"})
# 액터3애니메이션
pandaActor3.loop("walk3")


#액터1
lMinPt, lMaxPt = p3c.Point3(), p3c.Point3()
pandaActor.calcTightBounds(lMinPt, lMaxPt) # in the object space
print(lMinPt, lMaxPt)
m_os2ls = p3c.LMatrix4.scaleMat(5, 5, 5)
trans = p3c.LMatrix4.translateMat(0,0,0) 
pandaActor.setMat(trans*m_os2ls) 
pandaActor.reparentTo(base.render)
pandaActor.setR(180)

#액터2
lMinPt2, lMaxPt2 = p3c.Point3(), p3c.Point3()
pandaActor2.calcTightBounds(lMinPt2, lMaxPt2) # in the object space
print(lMinPt2, lMaxPt2)
m_os2ls2 = p3c.LMatrix4.scaleMat(1/20, 1/20, 1/20)
trans2 = p3c.LMatrix4.translateMat(0,0,0) 
#rotate = p3c.LMatrix4.rotateMat()    # 그냥 예비용으로 만들어둔거
example = p3c.LMatrix4()
example.multiply(m_os2ls,trans)
pandaActor2.setMat(example)             # *************multiply 예시로 만들어둠
pandaActor2.reparentTo(base.render)

#액터3
lMinPt3, lMaxPt3 = p3c.Point3(), p3c.Point3()
pandaActor3.calcTightBounds(lMinPt3, lMaxPt3) # in the object space
print(lMinPt3, lMaxPt3)
m_os2ls2 = p3c.LMatrix4.scaleMat(5, 5, 5)
trans2 = p3c.LMatrix4.translateMat(0,0,0) 
pandaActor3.setMat(trans2*m_os2ls2) 
pandaActor3.reparentTo(base.render)


#랜더링 효과
# directionalLight = DirectionalLight('directionalLight')
# directionalLight.setColor((0.2, 0.2, 0.8, 1))
# directionalLightNP = base.render.attachNewNode(directionalLight)
# directionalLightNP.setHpr(0, -20, 0)
# pandaActor.setLight(directionalLightNP)
# pandaActor2.setLight(directionalLightNP)
# pandaActor3.setLight(directionalLightNP)
pandaActor.setTransparency(TransparencyAttrib.MAlpha)
pandaActor.setAlphaScale(0.5)
pandaActor2.setTransparency(TransparencyAttrib.MAlpha)
pandaActor2.setAlphaScale(0.5)
pandaActor3.setTransparency(TransparencyAttrib.MAlpha)
pandaActor3.setAlphaScale(0.5)

#라인
lines = LineSegs()
lines.setColor(255,0,0)
lines.moveTo(0,0,0)
lines.drawTo(10,0,0)
lines.setColor(0,255,0)
lines.moveTo(0,0,0)
lines.drawTo(0,10,0)
lines.setColor(0,0,255)
lines.moveTo(0,0,0)
lines.drawTo(0,0,10)
lines.setThickness(10)
node = lines.create()
n = NodePath(node)
n.reparentTo(pandaActor)
n.setAlphaScale(2)

lines2 = LineSegs()
lines2.setColor(255,0,0)
lines2.moveTo(0,0,0)
lines2.drawTo(5,0,0)
lines2.setColor(0,255,0)
lines2.moveTo(0,0,0)
lines2.drawTo(0,5,0)
lines2.setColor(0,0,255)
lines2.moveTo(0,0,0)
lines2.drawTo(0,0,5)
lines2.setThickness(10)
node2 = lines2.create()
n2 = NodePath(node2)
n2.reparentTo(pandaActor2)
n2.setAlphaScale(2)

lines3 = LineSegs()
lines3.setColor(255,0,0)
lines3.moveTo(0,0,0)
lines3.drawTo(80,0,0)
lines3.setColor(0,255,0)
lines3.moveTo(0,0,0)
lines3.drawTo(0,80,0)
lines3.setColor(0,0,255)
lines3.moveTo(0,0,0)
lines3.drawTo(0,0,80)
lines3.setThickness(10)
node3 = lines3.create()
n3 = NodePath(node3)
n3.reparentTo(pandaActor3)
n3.setAlphaScale(2)

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
square_size = 40.0 # mm unit
pattern_points *= square_size

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)
arucoParams = cv.aruco.DetectorParameters_create()
matView=[]

def change1(df):#lmatrix4f로 변환
    return p3c.LMatrix4f(
        df[0][0],df[0][1],df[0][2],df[0][3],
        df[1][0],df[1][1],df[1][2],df[1][3],
        df[2][0],df[2][1],df[2][2],df[2][3],
        df[3][0],df[3][1],df[3][2],df[3][3])
    

def updateBg(task):
    global T1
    global T1_inv
    global T2
    global matView

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
        textObject.setText("No AR Marker Detected")
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        pandaActor.hide()
        pandaActor2.hide()
        pandaActor3.hide()
        
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

                    # convert each of the (x, y)-coordinate pairs to integers
                    axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                                                    intrisic_mtx, None)

                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrisic_mtx, None)

                    draw(frame, tuple(topLeft.ravel().astype(int)), imgpts)
                    
                    #중앙좌표
                    topLeft[0]=(topLeft[0]+topRight[0]+bottomRight[0]+ bottomLeft[0])/4
                    topLeft[1]=(topLeft[1]+topRight[1]+bottomRight[1]+ bottomLeft[1])/4
                    topRight[0]=(bottomRight[0]+topRight[0])/2
                    topRight[1]=(bottomRight[1]+topRight[1])/2
                    bottomLeft[0]=(bottomRight[0]+bottomLeft[0])/2
                    bottomLeft[1]=(bottomRight[1]+bottomLeft[1])/2
                    
                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                        np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                            intrisic_mtx, None)
                    
                    matView = getViewMatrix(rvecs, tvecs)
                    
                    for i in range(4):
                        for j in range(4):
                            T1[i][j]=matView[i][j]
                                
                    matViewInv = matView
                    matViewInv.invertInPlace()
                    
                    for i in range(4):
                        for j in range(4):
                            T1_inv[i][j]=matViewInv[i][j]

                    
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
                    

                   
                    
                #마커2
                if markerID==2 and matView != None:
                    pandaActor2.show()
                    corners2 = markerCorner.reshape((4, 2))
                    (topLeft2, topRight2, bottomRight2, bottomLeft2) = corners2
                    # convert each of the (x, y)-coordinate pairs to integers
                    axis2 = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs2, tvecs2 = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft2, topRight2, bottomLeft2, bottomRight2]).reshape(-1, 2), 
                                                    intrisic_mtx, None)

                    imgpts2, jac = cv.projectPoints(axis2, rvecs2, tvecs2, intrisic_mtx, None)

                    draw(frame, tuple(topLeft2.ravel().astype(int)), imgpts2)
                      
                    topLeft2[0]=(topLeft2[0]+topRight2[0]+bottomRight2[0]+ bottomLeft2[0])/4
                    topLeft2[1]=(topLeft2[1]+topRight2[1]+bottomRight2[1]+ bottomLeft2[1])/4
                    topRight2[0]=(bottomRight2[0]+topRight2[0])/2
                    topRight2[1]=(bottomRight2[1]+topRight2[1])/2
                    bottomLeft2[0]=(bottomRight2[0]+bottomLeft2[0])/2
                    bottomLeft2[1]=(bottomRight2[1]+bottomLeft2[1])/2
                    
                    ret, rvecs2, tvecs2 = cv.solvePnP(pattern_points, 
                        np.asarray([topLeft2, topRight2, bottomLeft2, bottomRight2]).reshape(-1, 2), 
                            intrisic_mtx, None)
                    
                    
                    matView2 = getViewMatrix(rvecs2, tvecs2)
                    
                    T2_ORI=np.zeros((4, 4), np.float32)
                    for i in range(4):
                        for j in range(4):
                            T2_ORI[i][j]=matView2[i][j]
                            
                    
                    matViewInv2 = matView2
                    matViewInv2.invertInPlace()

                    for i in range(4):
                        for j in range(4):
                            T2[i][j]=matViewInv2[i][j]
                            
                    #LM4 = p3c.LMatrix4()
                    #T1=change1(T1)
                    #T2=change1(T2)
                    #LM4.multiply(T1, T2)
                    #m2=LM4.multiply(T1, T2)
                    
                    m2=np.dot(T1,T2)
                    #m2=np.dot(m2,T2)
                    m2=change1(m2)
                    
                    #m2=p3c.LMatrix4.multiply(matViewInv2, T1)
                    pandaActor2.setMat(m2)
                    pandaActor2.setScale(10,10,10)
                    pandaActor2.setR(180)
                    pos2=pandaActor2.getPos()
                    pos1=pandaActor.getPos()
                    pandaActor2.setPos(-1*pos2[0],pos1[1],0)

                    
                    
                #마커3
                if markerID==3 and matView != None:
                    pandaActor3.show()
                    corners3 = markerCorner.reshape((4, 2))
                    (topLeft3, topRight3, bottomRight3, bottomLeft3) = corners3
                    # convert each of the (x, y)-coordinate pairs to integers
                    axis3 = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs3, tvecs3 = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft3, topRight3, bottomLeft3, bottomRight3]).reshape(-1, 2), 
                                                    intrisic_mtx, None)

                    imgpts3, jac = cv.projectPoints(axis3, rvecs3, tvecs3, intrisic_mtx, None)

                    draw(frame, tuple(topLeft3.ravel().astype(int)), imgpts3)
                    
                    topLeft3[0]=(topLeft3[0]+topRight3[0]+bottomRight3[0]+ bottomLeft3[0])/4
                    topLeft3[1]=(topLeft3[1]+topRight3[1]+bottomRight3[1]+ bottomLeft3[1])/4
                    topRight3[0]=(bottomRight3[0]+topRight3[0])/2
                    topRight3[1]=(bottomRight3[1]+topRight3[1])/2
                    bottomLeft3[0]=(bottomRight3[0]+bottomLeft3[0])/2
                    bottomLeft3[1]=(bottomRight3[1]+bottomLeft3[1])/2
                    
                    ret, rvecs3, tvecs3 = cv.solvePnP(pattern_points, 
                        np.asarray([topLeft3, topRight3, bottomLeft3, bottomRight3]).reshape(-1, 2), 
                            intrisic_mtx, None)
                    
                    
                    matView3 = getViewMatrix(rvecs3, tvecs3)
                    matViewInv3 = matView3
                    matViewInv3.invertInPlace()

                    for i in range(4):
                        for j in range(4):
                            T3[i][j]=matViewInv3[i][j]
                            
                    m3=np.dot(T1,T3)
                    m3=change1(m3)
                    
   
                    #m2=p3c.LMatrix4.multiply(matViewInv2, T1)
                    pandaActor3.setMat(m3)
                    pandaActor3.setScale(1,1,1)
                    pandaActor3.setR(180)
                    pos3=pandaActor3.getPos()
                    pos1=pandaActor.getPos()
                    pandaActor3.setPos(-1*pos3[0],pos1[1],0)

                      
                
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
