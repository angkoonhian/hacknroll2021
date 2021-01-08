#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install pytesseract')
get_ipython().system('pip install PyPDF2')


# In[23]:


import cv2 
import pytesseract
import numpy as np


# In[24]:


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


# In[44]:


def read_image(file_name):
    img = cv2.imread(file_name)
    
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(img, config=custom_config)
    
    return text 


# In[45]:


read_image(r'C:\Users\alici\Downloads\Hackathons\Hack&Roll2021\test_image.png')


# In[31]:


import PyPDF2

def read_pdf(file_name):
  pdfFileObj = open(file_name, 'rb')

  # call and store PdfFileReader
  pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

  #  to access first page pass 0
  pageObj = pdfReader.getPage(0)

  # extract the page object
  #  by extractText() function

  text = " ".join(pageObj.extractText().split())

  return text


# In[18]:


print(read_pdf(r'C:\Users\alici\Downloads\Hackathons\Hack&Roll2021\test_text.pdf'))


# In[ ]:




