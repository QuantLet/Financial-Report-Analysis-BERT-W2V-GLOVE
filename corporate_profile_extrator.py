import PyPDF2
import re

def f_correction(text):
    # Correct the error about "fl" and "fi" made by pdf extractor
    text = text.replace('˚', 'fi')
    text = text.replace('˜', 'fl')
    text = text.replace('™', "'")
    return text

def get_corporate_profile(file_name):
    # Take the text in the cooperate profile page
    global corporate_profile
    pdf_file = open(file_name, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    for p in range(number_of_pages):
        page = read_pdf.getPage(p)
        page_content = page.extractText()
        page_content = f_correction(page_content)

        regex = r"corporate +proflle"
        match = re.search(regex, page_content.lower())
        # Seach for the first page with the word 'company profile'
        if match != None:
            # print(p)
            break
    return corporate_profile
