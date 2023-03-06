import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import qrcode
import os


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
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )

        #Create QR Code

        qr.add_data(self.id)
        qr.make(fit=True)
        qrimage = qr.make_image(fill_colour="black", back_colour="white")

        filename = self.id + ".png"
 
        qrimage.save(filename, format='PNG')
        

        if self.isBlank:

            #Draw artowkr box
            self.pdf.set_xy(29, 29)
            self.pdf.cell(152, 152, "", 1, 1, 'C')

            #Insert QR Code
            self.pdf.image(filename, 6, 6, 20, 20, type="png")
            self.pdf.set_line_width(0.8)

            #Draw Name
            self.pdf.set_xy(122, 242)
            self.pdf.cell(73, 13, self.name, 1, 1, 'C')

            #Draw Class
            self.pdf.set_xy(122, 260)
            self.pdf.cell(73, 13, self.classgroup, 1, 1, 'C')

            self.pdf.output('test.pdf')

            if os.path.exists(filename):
                os.remove(filename)

temp = Template("John Gallagher", "1st Class", "Tiermohan NS", "1000587649NGH", True)
temp.generate()




