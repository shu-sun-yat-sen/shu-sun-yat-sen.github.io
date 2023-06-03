import cv2
import numpy as np
import math
def detect_ball(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Gauss_img = cv2.GaussianBlur(lab_img, (3,3), 0)
    white_mask = cv2.inRange(Gauss_img, (220, 0, 0), (255, 250, 255))
    white_open = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    white_closed = cv2.morphologyEx(white_open, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    edge_detect = cv2.Canny(white_closed, 100, 300)
    circles = cv2.HoughCircles(edge_detect, cv2.HOUGH_GRADIENT, 1, int(max(white_closed.shape[0],white_closed.shape[1])/3), param1=50, param2=10, minRadius=10, maxRadius=50)
    r = None
    x = None
    y = None
    if not circles is None:
        circles = np.uint16(np.around(circles))
        r, x, y = circles[0, 0, 2], circles[0, 0, 0], circles[0, 0, 1]
    return r,x,y



def get_area_max_contour(contours, threshold):
    """
    获取面积最大的轮廓
    :param contours: 要比较的轮廓的列表
    :param threshold: 轮廓要求的最小面积阈值
    :return: 面积最大的轮廓, 面积最大的轮廓的面积值
    """
    contours = map(lambda x: (x, math.fabs(cv2.contourArea(x))), contours)  # 计算所有轮廓面积
    contours = list(filter(lambda x: x[1] > threshold, contours))
    if len(contours) > 0:
        return max(contours, key = lambda x: x[1])  # 返回最大的轮廓
    else:
        return None, 0


def detect_keng(img):
    org_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2LAB)  # 将图片转化为lab格式
    Imask = cv2.inRange(img, (25, 63, 120), (255, 123, 180) ) # 根据lab值对图片进行二值化，提取洞的周围
    Imask = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 3)
    Imask = cv2.morphologyEx(Imask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations = 7)
    stencil  = np.zeros(org_img.shape[:-1], np.uint8)   # 全为0的掩膜
    contours = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]    
    max_contour = get_area_max_contour(contours, 1000)[0]   # 找到最大轮廓
    cv2.fillPoly(stencil, [max_contour], 255)   # 将最大区域填充为白色，即将洞填充为白色
    result = cv2.bitwise_xor(Imask, stencil)    # 异或
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    max_contour_hole = get_area_max_contour(contours, 700)[0]  #找到最大轮廓

    rect = cv2.minAreaRect(max_contour_hole)   # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    box = np.int0(cv2.boxPoints(rect))  

    p = 0
    for i in range(4):
        if box[i][1] > box[p][1]:
            p = i
        q = 1
    for i in range(4):
        if i != p and box[i][1] >= box[q][1]:
            q = i
    if box[p][0] > box[p][1]:
        p,q = q,p

    r = math.sqrt(math.pow(box[p] - box[q], 2) + math.pow(box[p] - box[q], 2))
    return box[p][0],box[p][1],r