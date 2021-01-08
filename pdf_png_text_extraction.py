import cv2 
import pytesseract
import numpy as np
import PyPDF2
import os
import word_summarizer


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

def read_image(file_name):
    img = cv2.imread(file_name)
    
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(img, config=custom_config)
    
    return text 

def read_pdf(file_name):
    pdfFileObj = open(file_name, 'rb')

    # call and store PdfFileReader
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    # Get number of pages
    number_of_pages = pdfReader.getNumPages()
    
    text= ""
    for page_number in range(number_of_pages):
      
        page = pdfReader.getPage(page_number)
        page_content = page.extractText()
        text += " ".join(page_content.split()) + " "

    return text

def process_pdf(filestorage):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    filestorage.save(f"{curr_dir}/data/tmp.pdf")
    file_dir = f"{curr_dir}/data/tmp.pdf"
    text = read_pdf(file_dir)
    tokenizer, model = word_summarizer.load_model_tokenizer_BERT()
    sentence_list = word_summarizer.split_into_sentences(text)
    summary_length = len(sentence_list)//4
    if summary_length == 0:
        return ""
    summarized_text = word_summarizer.sentence_summarizer(sentence_list, tokenizer, model, int(summary_length))
    return summarized_text


