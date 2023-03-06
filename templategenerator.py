import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import qrcode
from io import BytesIO

class Template:

    pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
    name = ''
    classgroup = ''
    school = ''
    id = ''
    isBlank = True

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

    def generate(self):

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )

        buffer = BytesIO()

        qr.add_data(self.id)
        qr.make(fit=True)
        qrimage = qr.make_image(fill_colour="black", back_colour="white")

        qrimage.save(buffer)

        if self.isBlank:
            
            #set page background to universal box and wording and place qr code
            self.pdf.image('TemplatePrototype.png', x = 0, y = 0, w = 210, h = 297)
            self.pdf.image(buffer, 6, 6, 20, 20)

            self.pdf.set_xy(122, 242)
            self.pdf.cell(73, 13, self.name, 1, 1, 'C')

            self.pdf.set_xy(122, 260)
            self.pdf.cell(73, 13, self.classgroup, 1, 1, 'C')

            self.pdf.output('test.pdf')

temp = Template("John Gallagher", "1st Class", "Tiermohan NS", "1000587649NGH", True)
temp.generate()




