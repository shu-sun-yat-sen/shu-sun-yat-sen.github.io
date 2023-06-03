# coding=utf-8
import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("origin", 1)
flag = False
def draw_and_get_color(event,x,y,flags,param):
    global flag
    h,s,v = hsv[y,x]
    global img
    if event == cv2.EVENT_MOUSEMOVE:
        if flag:
            cv2.circle(img,(x,y),5,(3,3,254),-1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        img = img_copy.copy()
        flag = True
        txt = 'H: '+str(h)+' S: '+str(s)+' V: '+str(v)
        cv2.putText(img,txt,(5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.circle(img,(x,y),5,(3,3,254),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        print(112233322)
        flag = False
        mask = cv2.inRange(img,(2, 2, 253),(4, 4, 255))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        idx = np.argmax([cv2.contourArea(contours[i]) for i in range(len(contours))])
        cv2.drawContours(img,contours=contours,contourIdx=idx,color=(255,0,0),thickness=2)
        area = cv2.contourArea(contours[idx])
        lenth = cv2.arcLength(contours[idx],True)
        txt2 = 'area: ' + str(area) + ' clength: ' + str(lenth)
        cv2.putText(img,txt2,(5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.setMouseCallback('origin',draw_and_get_color)
img = cv2.imread('blade.webp')
img_copy = img
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    cv2.imshow("origin", img)
    key = cv2.waitKey(100)
    if key == 27:
        # 按esc键退出
        print("esc break...")
        break


cv2.destroyWindow("origin")