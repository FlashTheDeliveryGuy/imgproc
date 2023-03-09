import cv2
from pyzbar.pyzbar import decode
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image

from deskew import determine_skew
from typing import Tuple, Union
import math
import imutils
import time
from multiprocessing import Lock, Process, Queue, current_process
import queue # imported for using queue.Empty exception

dpmm = 11.811 #dots per mmm 
xyqrprint = 6
idealx = round(dpmm*xyqrprint)

global counter


class Template_Page:

    @staticmethod
    def imagePreProcess(self, image) -> np.array:

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #angle = determine_skew(imageGray)

        #imageGray = self.rotate(imageGray, angle)
        denoised = cv2.medianBlur(imageGray, 3)

      
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        image_sharp = cv2.filter2D(src=denoised, ddepth=-1, kernel=kernel)


        ret,th2 = cv2.threshold(image_sharp,150,255,cv2.THRESH_BINARY)

        return th2


    @staticmethod
    def arraySort(self, array) -> np.array:

        a = array
        dt = [('col1', a.dtype),('col2', a.dtype)]
        assert a.flags['C_CONTIGUOUS']
        b = a.ravel().view(dt)
        b.sort(order=['col2','col1'])

        return a
    
    @staticmethod
    def arraySortY(self, array) -> np.array:

        a = array
        dt = [('col1', a.dtype),('col2', a.dtype)]
        assert a.flags['C_CONTIGUOUS']
        b = a.ravel().view(dt)
        b.sort(order=['col2'])

        return a
    
    @staticmethod
    def arraySortX(self, array) -> np.array:

        a = array
        dt = [('col1', a.dtype),('col2', a.dtype)]
        assert a.flags['C_CONTIGUOUS']
        b = a.ravel().view(dt)
        b.sort(order=['col1'])

        return a
    
    @staticmethod
    def pointSelector(self, array, selector):
        
        a = array

        #try:
        self.arraySortY(self, a)
        
        tops = np.array([a[0], a[1]])
        bottoms = np.array([a[2],a[3]])

        self.arraySortX(self,tops)
        self.arraySortX(self,bottoms)

        if selector == 0:
            return tops[0]
        elif selector == 1:
            return tops[1]
        elif selector == 2:
            return bottoms[0]
        else:
            return bottoms[1]

    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:

        background = (0,0,0)
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2

        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    @staticmethod
    def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):

        # convert both the input image and template to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # use ORB to detect keypoints and extract (binary) local
        # invariant features
        orb = cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)

        # sort the matches by their distance (the smaller the distance,
        # the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)

        # keep only the top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]

        # check to see if we should visualize the matched keypoints
        if debug:
            matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                matches, None)
            matchedVis = imutils.resize(matchedVis, width=1000)
            cv2.imshow("Matched Keypoints", matchedVis)
            cv2.waitKey(0)

        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")

        # loop over the top matches
        for (i, m) in enumerate(matches):

            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

        # compute the homography matrix between the two sets of matched
        # points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(image, H, (w, h))

        # return the aligned image
        return aligned
    
    results = ''
    document = ''
    image = ''
    points = np.array([])
    dimensions = (2480, 3508)

    
    def __init__(self, pdfpage):


        
        #convert page to a PyMuPDF pixmap
        pix = pdfpage.get_pixmap(dpi = 300, colorspace = "RGB")
        

        
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        self.image =bytes.reshape(pix.height, pix.width, pix.n)

        #first QR scanning

        qcd = cv2.QRCodeDetector()
      
        self.setPointData(qcd)

        if self.pointID.sum() > self.pointC.sum():
            self.image = cv2.flip(self.image,-1)

        self.setPointData(qcd)


    def setPointData(self, qcd):

        retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(self.image)

        if len(decoded_info) < 4:

            retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(self.imagePreProcess(self, self.image))

            if len(decoded_info) < 4:
                raise Exception('Invalid template')

        for name,points in zip(decoded_info, points):
            if name == '0000':
                self.pointA = points

            elif name == '0001':
                self.pointB = points

            elif name == '0002':
                self.pointC = points 

            else:
                self.pointID = points
                self.ID = name       
        
    def crop(self):    

        #x = self.pointSelector(self, self.pointID, 0)
        

        x1y1 = (self.pointSelector(self, self.pointID, 0))
        x1y2 = (self.pointSelector(self, self.pointC, 2))
        x2y1 = (self.pointSelector(self, self.pointA, 1))
        x2y2 = (self.pointSelector(self, self.pointB, 3))

        print(x1y1, x1y2, x2y1, x2y2)              

        newTransform= (x1y1, x2y1, x1y2, x2y2)

        sourceCoordList=[(100,100), (60,3450), (2423,57), (2423,3450)]

        sourceCoordList = sorted(sourceCoordList , key=lambda k: [k[1], k[0]])

        qrCoordList = np.array(sorted(newTransform , key=lambda k: [k[1], k[0]]))
        sourceCoordList = np.array(sourceCoordList)

        h, status = cv2.findHomography(qrCoordList, sourceCoordList)

        warpedImage = cv2.warpPerspective(self.image, h, (self.image.shape[1], self.image.shape[0]))

        croppedimage = warpedImage[361:2120, 361:2120]

        plt.imshow(croppedimage)
        plt.show()

        return croppedimage
    
    def getID(self):

        return self.ID
      
#generated file should have qr code places at 100,100 with a width and height of 178,178

#doc = fitz.open('testdocuments/scantest2.pdf')

#for page in doc:

    #print('scanning and cropping page ')

    #temp = Template_Page(page)
    #crop = temp.crop()

    #plt.imshow(crop)
    #plt.show()

#temp = QRScanner('testdocuments/scan10002.pdf')
#temp = QRScanner('test.pdf')
#temp2 = QRScanner('test.pdf')
#temp2 = temp.deskew()
#temp.crop()
