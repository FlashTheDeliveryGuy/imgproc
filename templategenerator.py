import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import qrcode
import os

dpmm = 11.811 #dots per mmm 


class Template:

    pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
    name = ''
    classgroup = ''
    school = ''
    id = ''
    isBlank = True
    title = 'Draw your picture inside the lines.'
    warningMessage1 = 'Any damage to the alignment marks or QR code will render'
    warningMessage2 = 'this template void and we cannot be held responsible for'
    warningMessage3 = 'any issues this may cause.'

    def __init__(self, name, classgroup, school, id, isBlank):

        self.name = name
        self.classgroup = classgroup
        self.school = school
        self.id = id
        self.isBlank = isBlank 

        self.pdf.add_font('Roboto-Bold','','./Roboto-Bold.ttf',uni=True)
        self.pdf.add_page()
        self.pdf.set_font('Roboto-Bold', '', 10)
        self.pdf.set_text_color(0, 0, 0)

        #self.pdf.set_xy(30, 225)
        #self.pdf.cell(113, 15, self.warningMessage, 0, 0, 'L')

        self.pdf.set_line_width(2)

    def generate(self):

        #QR Code Settings
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=20,
            border=4,
        )

        qrA = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=20,
            border=4,
        )

        qrB = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=20,
            border=4,
        )

        qrC = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=20,
            border=4,
        )



        #Create QR Code

        qr.add_data(self.id)
        qr.make(fit=True)
        qrimage = qr.make_image(fill_colour="black", back_colour="white")

        qrA.add_data('0000')
        qrA.make(fit=True)
        qrimageA = qrA.make_image(fill_colour="black", back_colour="white")

        qrB.add_data('0001')
        qrB.make(fit=True)
        qrimageB = qrB.make_image(fill_colour="black", back_colour="white")

        qrC.add_data('0002')
        qrC.make(fit=True)
        qrimageC = qrC.make_image(fill_colour="black", back_colour="white")



        filename = self.id + ".png"
 
        qrimage.save(filename, format='PNG')

        filenameA = "crossMark.png"

        qrimageA.save(filenameA, format='PNG')

        filenameB = "CircleMark.png"

        qrimageB.save(filenameB, format='PNG')

        filenameC = "squareMark.png"

        qrimageC.save(filenameC, format='PNG')
        

        if self.isBlank:

            #Draw artwork box
            self.pdf.set_xy(29, 29)
            self.pdf.cell(152, 152, "", 1, 1, 'C')

            #Insert QR Code
            self.pdf.image(filename, 6, 6, 20, 20, type="png")
            self.pdf.set_line_width(0.8)

            self.pdf.image('square.png', 194, 281, 10, 10, type="png")

            self.pdf.image('circle.png', 194, 6, 10, 10, type="png")

            self.pdf.image('cross.png', 6, 281, 10, 10, type="png")

            #Draw Name
            self.pdf.set_xy(122, 242)
            self.pdf.cell(73, 13, '', 1, 1, 'C')

            #Draw Class
            self.pdf.set_xy(122, 260)
            self.pdf.cell(73, 13, '', 1, 1, 'C')

            self.pdf.output('test.pdf')

            if os.path.exists(filename):
                os.remove(filename)

            if os.path.exists(filenameA):
                os.remove(filenameA)
            if os.path.exists(filenameB):
                os.remove(filenameB)
            if os.path.exists(filenameC):
                os.remove(filenameC)

temp = Template("John Gallagher", "1st Class", "Tiermohan NS", "1000587649NGH", True)
temp.generate()




