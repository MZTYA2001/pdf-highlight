from calendar import c
import re
import tempfile
from bs4 import BeautifulSoup
from openai import OpenAI
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
import fitz
import os
import pdfplumber
from symbol import term
import PyPDF2


class PDFUploader:
    def __init__(self):
        self.pdf_file = None
        self.text = ""

    def upload_pdf(self):
        # Upload a PDF file
        self.pdf_file = st.file_uploader("Upload your PDF", type='pdf')
        return self.pdf_file

    def process_pdf(self):
        # Process the uploaded PDF file
        if self.pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(self.pdf_file)
            for page in pdf_reader.pages:
                self.text += page.extract_text()
            return self.text

class PDFContextExtractor:
    def __init__(self, pdf_file, model_name="gpt-3.5-turbo", temperature=0):
        self.pdf_file = pdf_file
        self.model_name = model_name
        self.temperature = temperature
        self.vectorstore = None
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.last_context = ""

    def load_and_index_pdf(self):
        if self.pdf_file is None:
            return
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(self.pdf_file.getbuffer())
            tmp_pdf_path = tmpfile.name

        # Use the temporary file path for the loader
        loader = UnstructuredPDFLoader(tmp_pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )

    def format_docs_with_context(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_context(self, query):
        retriever = self.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        self.last_context = self.format_docs_with_context(docs)
        return docs

    def invoke(self, question):
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": self.retrieve_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)

    def cleanup(self):
        if self.vectorstore:
            self.vectorstore.delete_collection()


class PDFSearchAndDisplay:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(self.pdf_path)
        self.context_pages = []

    def find_highlight_and_screenshot_context(self, pdf_path, context, output_folder):
        # Delete any existing folder with screenshots
        if os.path.exists(output_folder):
            for file_name in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(output_folder)
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Split context into a minimum of atleast three words
        chunks = context.strip().split()
        if len(chunks) < 5:
            chunks = context.strip().split() + [""] * (5 - len(chunks))

        chunk_index = 0  # Index to keep track of the current chunk
        total_highlights = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    with fitz.open(pdf_path) as doc:
                        fitz_page = doc.load_page(page_num - 1)
                        page_highlighted = (
                            False  # Flag to track if any highlight is done on this page
                        )

                        while chunk_index < len(chunks):
                            chunk = chunks[chunk_index]
                            if chunk in text:
                                chunk_instances = fitz_page.search_for(chunk)
                                for inst in chunk_instances:
                                    fitz_page.add_highlight_annot(inst)
                                    page_highlighted = True
                                    total_highlights += 1
                                chunk_index += 1  # Move to the next chunk
                            else:
                                break  # If chunk not found, check on the next page

                        # Save the page if significant highlighting was done
                        if page_highlighted and total_highlights >= 20:
                            pix = fitz_page.get_pixmap()
                            output_file = os.path.join(
                                output_folder, f"page_{page_num}.png"
                            )
                            pix.save(output_file)
                            print(
                                f"Saved highlighted screenshot of page {page_num} to {output_file}"
                            )

                    # Reset chunk index if the end of context is reached
                    if chunk_index >= len(chunks):
                        chunk_index = 0  # Reset chunk index for the next iteration
                        break  # Exit if the end of the context is reached


client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# Function to create a label using OpenAI's GPT-3.5-Turbo model
def create_label(pdf_name):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Create a label for {pdf_name}."},
        ],
        stop=["\n"],
    )
    print(response)
    return response.choices[0].message.content


# Streamlit app
st.title("GPT-4-PDF Demo")
pdf_uploader = PDFUploader()
uploaded_pdf = pdf_uploader.upload_pdf()

if uploaded_pdf is not None:
    if 'label' not in st.session_state or uploaded_pdf.name != st.session_state.get('pdf_name', ''):
        st.session_state.label = create_label(uploaded_pdf.name)
        st.session_state.pdf_name = uploaded_pdf.name

    pdf_uploader.process_pdf()
    pdf_extractor = PDFContextExtractor(uploaded_pdf, "gpt-3.5-turbo", 0)
    pdf_extractor.load_and_index_pdf()

    user_input = st.text_input(f"Ask a question about {st.session_state.label}:") if st.session_state.label else st.text_input("Ask a question:")

    if user_input:
        answer = pdf_extractor.invoke(user_input)
        context = pdf_extractor.last_context

        # Display the answer
        st.write(f"Answer: {answer}")

        find = f"""{context}"""

        # Collapsible section for screenshots
        with st.expander("Screenshots", expanded=False):
            pdf_search = PDFSearchAndDisplay(uploaded_pdf)
            pdf_search.find_highlight_and_screenshot_context(
                uploaded_pdf, find, "screenshots"
            )

            # Dynamically list the files in the screenshots directory
            screenshot_files = [f for f in os.listdir("screenshots")]
            for file_name in screenshot_files:
                file_path = os.path.join("screenshots", file_name)
                if os.path.exists(file_path):
                    st.image(file_path)

        # Collapsible section for context
        with st.expander("Context", expanded=False):
            st.write(context)

    # Cleanup
    pdf_extractor.cleanup()
