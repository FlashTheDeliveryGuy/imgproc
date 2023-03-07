import cv2
from pyzbar.pyzbar import decode
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image

from deskew import determine_skew
from typing import Tuple, Union
import math

dpmm = 11.811 #dots per mmm 
xyqrprint = 6
idealx = round(dpmm*xyqrprint)

class QRScanner:

    @staticmethod
    def arraySort(self, array) -> np.array:

        a = array
        dt = [('col1', a.dtype),('col2', a.dtype)]
        assert a.flags['C_CONTIGUOUS']
        b = a.ravel().view(dt)
        b.sort(order=['col1','col2'])

        return a
    
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


    results = ''
    document = ''
    image = ''
    points = np.array([])
    referencePoints = np.array([[100,100],[100,277], [277,100], [278,278]], np.float32)
    dimensions = (2480, 3508)

    
    def __init__(self, pdf):

        self.document = pdf

        images = convert_from_path(self.document, size = (2480, 3508))
        self.image = np.array(images[0])

        #decode the QR code and store results to class variables

        results = decode(self.image)

        self.points = np.array(results[0].polygon, np.float32)

        #if the QR code indicates that it is upside down, rotate the entire image 180 deg.

        if results[0].orientation == 'DOWN':
            #self.image = self.image[::-1,::-1] #rotate the array 180 degrees
            self.image = cv2.flip(self.image,-1)

        cv2.imshow('window', self.image)
        cv2.waitKey(0)

    def updateQrPoints(self, image):

        print (image)

        results = decode(image)

        print(results)

        points = np.array(results[-1].polygon, np.float32)

        return points
    
    def crop(self):

        image = self.image
        

        decoderImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thr, decoderImage = cv2.threshold(decoderImage,230,255,cv2.THRESH_BINARY)

        angle = determine_skew(decoderImage)

        decoderImage = self.rotate(decoderImage, angle)
        image = self.rotate(image, angle)
        
        results = decode(decoderImage)

        x1y1 = ''
        x1y2 = ''
        x2y1 = ''
        x2y2 = ''

        for codes in results:

            print(codes.data)

            a = (codes.polygon[0].x,codes.polygon[0].y)
            b = (codes.polygon[1].x,codes.polygon[1].y)
            c = (codes.polygon[2].x,codes.polygon[2].y)
            d = (codes.polygon[3].x,codes.polygon[3].y)

            vertices = (a,b,c,d)
            vertices = sorted(vertices , key=lambda k: [k[1], k[0]])

            print(vertices)
         
            if codes.data == b'0000':

                print('0000')
                
                x2y1 = vertices[1]

            elif codes.data == b'0001':

                print('0001')

                x2y2 = vertices[3]

            elif codes.data == b'0002':

                print('0002')

                x1y2 = vertices[2]

            else:
                
                print('main')
                x1y1 = vertices[0]
                

        newTransform= (x1y1, x2y1, x1y2, x2y2)

        print(newTransform)


        sourceCoordList=[(100,100), (60,3450), (2423,57), (2423,3450)]

        sourceCoordList = sorted(sourceCoordList , key=lambda k: [k[1], k[0]])

        qrCoordList = np.array(sorted(newTransform , key=lambda k: [k[1], k[0]]))
        sourceCoordList = np.array(sourceCoordList)

        h, status = cv2.findHomography(qrCoordList, sourceCoordList)

        warpedImage = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
        cv2.rectangle(warpedImage,(0,0),(2480,3508),(0,255,0),2)
        cv2.rectangle(warpedImage,(361,361),(2120,2120),(0,255,0),2)
                                          
        cv2.imshow('window2', warpedImage)
        cv2.waitKey(0)

        return
      
#generated file should have qr code places at 100,100 with a width and height of 178,178


temp = QRScanner('testdocuments/scan10002.pdf')
#temp = QRScanner('test.pdf')
#temp2 = QRScanner('test.pdf')
#temp2 = temp.deskew()
temp.crop()
