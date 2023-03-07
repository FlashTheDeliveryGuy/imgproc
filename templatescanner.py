import cv2
from pyzbar.pyzbar import decode
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

dpmm = 11.811 #dots per mmm 
xyqrprint = 6
idealx = round(dpmm*xyqrprint)
print(idealx)

class QRScanner:

    def arraySort(self, array) -> np.array:

        a = array
        dt = [('col1', a.dtype),('col2', a.dtype)]
        assert a.flags['C_CONTIGUOUS']
        b = a.ravel().view(dt)
        b.sort(order=['col1','col2'])

        return a


    results = ''
    document = ''
    image = ''
    points = np.array([])
    referencePoints = np.array([[100,100],[100,277], [277,100], [278,278]], np.float32)
    dimensions = (2480, 3508)

    
    def __init__(self, pdf):

        self.document = pdf

        images = convert_from_path(self.document, poppler_path = r"C:\Users\Philip\Documents\GitHub\poppler-23.01.0\Library\bin", dpi=300)
        self.image = np.array(images[0])

        results = decode(self.image)

        self.points = np.array(results[0].polygon, np.float32)

       # for result in results:
           # print(result.type, result.data, result.quality, result.polygon)

    def updateQrPoints(self, image):

        results = decode(image)

        points = np.array(results[0].polygon, np.float32)

        return points

    def deskew(self):

        self.points = self.arraySort(self.points)

        angle = cv2.minAreaRect(self.points)[-1]
        
        if angle < 45:
            angle = angle
    
        else:
            angle = -angle     

        #plt.subplot(121),plt.imshow(self.image),plt.title('Input')

        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
      
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        print(M)
        rotated = cv2.warpAffine(self.image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
      
        #plt.subplot(122),plt.imshow(rotated),plt.title('Output')
        #plt.show()

        newQRpoints = self.updateQrPoints(rotated)

        return rotated
    
   # def extractArtwork(self):





temp = QRScanner('testdocuments/testpdf-1.pdf')
#temp2 = QRScanner('test.pdf')
temp2 = temp.deskew()
