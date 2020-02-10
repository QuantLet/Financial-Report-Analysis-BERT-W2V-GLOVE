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
    pdf_file = open(file_name, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    CPregex = r"corporate[\s{,3}|\n]pro"
    # TCregex = r"table[\s{,3}|\n]of[\s{,3}|\n]content"
    TCregex = r"content"
    corporate_profile = None
    for p in range(number_of_pages):
        page = read_pdf.getPage(p)
        page_content = page.extractText()
        page_content = f_correction(page_content)

        CPmatch = re.search(CPregex, page_content.lower())
        TCmatch = re.search(TCregex, page_content.lower())

        # Seach for the first page with the word 'company profile which is not table of content'
        if TCmatch == None:
            if CPmatch != None: # need to check whether there is enough content
                # print(p)
                corporate_profile = page_content
                break

    return corporate_profile
