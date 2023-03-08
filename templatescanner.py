import cv2
from pyzbar.pyzbar import decode
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image

from deskew import determine_skew
from typing import Tuple, Union
import math
import fitz
import imutils

import multiprocessing

dpmm = 11.811 #dots per mmm 
xyqrprint = 6
idealx = round(dpmm*xyqrprint)

global counter


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
    referencePoints = np.array([[100,100],[100,277], [277,100], [278,278]], np.float32)
    dimensions = (2480, 3508)

    
    def __init__(self, pdfpage):

        #align = self.align_images(img, template, debug=True)

        
        #convert page to a PyMuPDF pixmap
        pix = pdfpage.get_pixmap(dpi = 300, colorspace = "RGB")

        #cast the pixmap to an OpenCV compatible array.
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        self.image =bytes.reshape(pix.height, pix.width, pix.n)

        template = convert_from_path('test.pdf', dpi=300)
        template = np.asarray(template[0])

        align = self.align_images(self.image, template, debug=True)
        plt.imshow(align)
        plt.show()

        #first QR scanning
        results = decode(self.image)

        self.points = np.array(results[0].polygon, np.float32)

        #if the QR code indicates that it is upside down, rotate the entire image 180 deg.

        if results[0].orientation == 'DOWN':
            self.image = cv2.flip(self.image,-1)


    def updateQrPoints(self, image):

        print (image)

        results = decode(image)

        print(results)

        points = np.array(results[-1].polygon, np.float32)

        return points
    
    def crop(self):

        image = self.image

        qcd = cv2.QRCodeDetector()
        

        decoderImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #decoderImage = cv2.GaussianBlur(decoderImage, (2,2), 1)
        thr, decoderImage = cv2.threshold(decoderImage,200,255,cv2.THRESH_BINARY)

        angle = determine_skew(decoderImage)

        decoderImage = self.rotate(decoderImage, angle)
        image = self.rotate(image, angle)
        
        results = decode(decoderImage)

        #retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(decoderImage)
        #print(retval, decoded_info, points, straight_qrcode)

        x1y1 = ''
        x1y2 = ''
        x2y1 = ''
        x2y2 = ''

        decoderImageConvert = cv2.cvtColor(decoderImage, cv2.COLOR_BGR2RGB)
        plt.imshow(decoderImageConvert)
        plt.show()

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
        #cv2.rectangle(warpedImage,(0,0),(2480,3508),(0,255,0),2)
        #cv2.rectangle(warpedImage,(361,361),(2120,2120),(0,255,0),2)
                                          
        #cv2.imsave('window2', warpedImage)
        #cv2.waitKey(0)
        global counter
        croppedimage = warpedImage[361:2120, 361:2120]
        cv2.imwrite("crops/Cropped Image " + str(counter) + ".png", croppedimage)
        counter = counter + 1

        return
      
#generated file should have qr code places at 100,100 with a width and height of 178,178

doc = fitz.open('testdocuments/newmarks.pdf')
counter = 0
i = 0


for page in doc:

    i = i + 1

    print('scanning and cropping page ' + str(i) )

    temp = QRScanner(page)
    temp.crop()

#temp = QRScanner('testdocuments/scan10002.pdf')
#temp = QRScanner('test.pdf')
#temp2 = QRScanner('test.pdf')
#temp2 = temp.deskew()
#temp.crop()
