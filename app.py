import streamlit as st
import groq
import langchain
import fitz  # PyMuPDF
import pdfplumber
from langchain import LLMChain

class PDFContextExtractor:
    def __init__(self, groq_api_key, model_name):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        self.chain = LLMChain(llm=self.llm)

    def extract_context(self, pdf_path, query):
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        context = self.chain.run(input={"pdf_content": pdf_content, "query": query})
        return context


class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots


# Streamlit App
st.title("PDF Chat with Context Extraction and Highlighting")

groq_api_key = st.text_input("Enter your Groq API Key")
model_name = st.text_input("Enter the Groq Model Name", "gemma2-9b-it")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and groq_api_key and model_name:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    query = st.text_input("Enter your context query")
    search_term = st.text_input("Enter the search term for highlighting")
    
    if st.button("Extract Context"):
        extractor = PDFContextExtractor(groq_api_key=groq_api_key, model_name=model_name)
        context = extractor.extract_context(pdf_path, query)
        st.write("Context Extracted:")
        st.write(context)
    
    if st.button("Search and Highlight"):
        searcher = PDFSearchAndDisplay()
        highlighted_pages = searcher.search_and_highlight(pdf_path, search_term)
        screenshots = searcher.capture_screenshots(pdf_path, highlighted_pages)
        st.write("Highlighted Pages and Screenshots:")
        for page, screenshot in zip(highlighted_pages, screenshots):
            st.write(f"Page {page[0]}: {page[1]}")
            st.image(screenshot)

