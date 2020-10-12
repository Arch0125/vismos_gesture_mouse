import cv2
import numpy as np

from pynput.mouse import Button, Controller
import wx

ox, oy, ow, oh = 0, 0, 0, 0

# pink
lb3 = np.array([130, 80, 2])
ub3 = np.array([170, 255, 255])

# yellow
lb2=np.array([20,100,20])
ub2=np.array([40,255,255])

# green

lb1 = np.array([33, 80, 40])
ub1 = np.array([102, 255, 255])

# blue
lb = np.array([94, 80, 2])
ub = np.array([126, 255, 255])

msingle = 0
cn1 = 0
cn2 = 0

mLocOld = np.array([0, 0])
mouseLocLoc = np.array([0, 0])
df = 3

mouse = Controller()
app = wx.App(False)
(sx, sy) = wx.GetDisplaySize()
(camx, camy) = (320, 240)

cam = cv2.VideoCapture(1)
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

kernelOpen1 = np.ones((5, 5))
kernelClose1 = np.ones((20, 20))

kernelOpen2 = np.ones((5, 5))
kernelClose2 = np.ones((20, 20))

kernelOpen3 = np.ones((5, 5))
kernelClose3 = np.ones((20, 20))

while True:
    ret, img = cam.read()
    img = cv2.resize(img, (340, 220))

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lb, ub)
    mask1 = cv2.inRange(imgHSV, lb1, ub1)
    mask2 = cv2.inRange(imgHSV, lb2, ub2)
    mask3 = cv2.inRange(imgHSV, lb3, ub3)
    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskOpen1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernelOpen1)
    maskClose1 = cv2.morphologyEx(maskOpen1, cv2.MORPH_CLOSE, kernelClose1)

    maskOpen2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernelOpen2)
    maskClose2 = cv2.morphologyEx(maskOpen2, cv2.MORPH_CLOSE, kernelClose2)

    maskOpen3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernelOpen3)
    maskClose3 = cv2.morphologyEx(maskOpen3, cv2.MORPH_CLOSE, kernelClose3)

    maskFinal = maskClose
    maskFinal1 = maskClose1
    maskFinal2 = maskClose2
    maskFinal3 = maskClose3

    conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    conts1, h1 = cv2.findContours(maskFinal1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    conts2, h2 = cv2.findContours(maskFinal2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    conts3, h3 = cv2.findContours(maskFinal3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(img,conts,-1,(255,0,0),3)
    if len(conts) == 2:
        cn1 = 1
        cn2 = 1
        if (msingle == 1):
            msingle = 0
            mouse.release(Button.left)

        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
        cx1 = x1 + w1 // 2
        cy1 = y1 + h1 // 2
        cx2 = x2 + w2 // 2
        cy2 = y2 + h2 // 2
        cx = (cx1 + cx2) // 2
        cy = (cy1 + cy2) // 2

        # cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)
        # cv2.circle(img,(cx,cy),2,(0,0,255),2)

        mouseLoc = mLocOld + ((cx, cy) - mLocOld) / df
        mouse.position = (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy)

        while mouse.position == (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy):
            pass
        mLocOld = mouseLoc
        ox, oy, ow, oh = cv2.boundingRect(np.array([[[x1, y1], [x1 + w1, y1 + h1], [x2, y2], [x2 + w2, y2 + h2]]]))
        # cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
    elif (len(conts) == 1):

        x, y, w, h = cv2.boundingRect(conts[0])
        if (msingle == 0):
            if (abs((w * h - ow * oh) / (w * h)) < 0.4):
                msingle = 1
                mouse.press(Button.left)
                ox, oy, ow, oh = 0, 0, 0, 0

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(img, (cx, cy), (w + h) // 4, (0, 0, 255), 2)

        mouseLoc = mLocOld + ((cx, cy) - mLocOld) / df
        mouse.position = (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy)

        while mouse.position == (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy):
            pass
        mLocOld = mouseLoc
        # mouse.press(Button.left)

    if ((len(conts1) == 1) and len(conts) == 1):

            x3, y3, w3, h3 = cv2.boundingRect(conts1[0])
            cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)
            cx = x3 + w3 // 2
            cy = y3 + h3 // 2
            if (cn1 == 1):
                mouse.click(Button.right, 1)
            cn1 = 0

    if (len(conts3) == 1) and (len(conts) == 1):
        if (cn2 == 1):
            x4, y4, w4, h4 = cv2.boundingRect(conts3[0])
            cv2.rectangle(img, (x4, y4), (x4 + w4, y4 + h4), (255, 0, 0), 2)
            cx = x4 + w4 // 2
            cy = y4 + h4 // 2

            mouse.click(Button.left, 2)
            cn2 = 0

    if len(conts2) == 1 and ((len(conts) == 1) or (len(conts) == 2)):
        x5, y5, w5, h5 = cv2.boundingRect(conts2[0])
        cv2.rectangle(img, (x5, y5), (x5 + w5, y5 + h5), (255, 0, 0), 2)
        cx = x5 + w5 // 2
        cy = y5 + h5 // 2

        if (len(conts) == 1):
            mouse.scroll(0, -1)
        if (len(conts) == 2):
            mouse.scroll(0, 1)

    # cv2.imshow("maskClose",maskClose)
    # cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("cam", img)
    cv2.waitKey(10)
