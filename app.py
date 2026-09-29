import time
import streamlit as st
import fitz  # PyMuPDF
import os
from google import genai


# -----------------------------
# Gemini configuration
# -----------------------------
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# -----------------------------
# Generate document summary
# -----------------------------
def generate_summary(document_text):

    prompt = f"""
You are ASTRA INTEL, a document intelligence assistant.

Summarize ONLY the document provided below.
Do not use outside knowledge.
Do not invent information.

Give the summary in a clear and concise format with:

1. Main purpose of the document
2. Key points
3. Important conclusions or findings

DOCUMENT:
{document_text}
"""

    # Try up to 3 times if Gemini is temporarily unavailable
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt
            )

            return response.text

        except Exception as e:

            if "503" in str(e) and attempt < 2:

                delay = 5 * (2 ** attempt)

                st.info(
                    f"Gemini is temporarily busy. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            else:

                raise e


# -----------------------------
# Generate answer
# -----------------------------
def generate_answer(document_text, question):

    prompt = f"""
You are ASTRA INTEL, a document intelligence assistant.

Answer the user's question using ONLY the document provided below.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer cannot be found in the document, say:
  "The answer is not available in the uploaded document."
- Give a clear and concise answer.
- Mention the relevant page number when possible.

DOCUMENT:
{document_text}

USER QUESTION:
{question}
"""

    # Try up to 3 times if Gemini is temporarily unavailable
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt
            )

            return response.text

        except Exception as e:

            if "503" in str(e) and attempt < 2:

                delay = 5 * (2 ** attempt)

                st.info(
                    f"Gemini is temporarily busy. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            else:

                raise e


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ASTRA INTEL",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------
st.title("🛡️ ASTRA INTEL")

st.write(
    "AI-powered defence document intelligence system "
    "for document summarization and grounded question answering."
)

st.divider()


# -----------------------------
# PDF Upload
# -----------------------------
st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file is not None:

    try:

        # Read uploaded PDF
        pdf_bytes = uploaded_file.read()

        # Open PDF from memory
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        st.success(
            f"Successfully loaded: {uploaded_file.name}"
        )

        st.write(
            f"**Number of pages:** {len(document)}"
        )

        # Store pages and their text
        pages = []

        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = page.get_text("text").strip()

            pages.append({
                "page": page_number,
                "text": text
            })


        # -----------------------------
        # Generate Summary
        # -----------------------------
        st.header("2. Document Summary")

        if st.button("Generate Summary"):

            # Combine all extracted pages
            document_text = "\n\n".join(
                f"Page {page['page']}:\n{page['text']}"
                for page in pages
            )

            if not document_text.strip():

                st.warning(
                    "No readable text was found in this PDF."
                )

            else:

                with st.spinner(
                    "Generating document summary..."
                ):

                    try:

                        summary = generate_summary(
                            document_text
                        )

                        st.write(summary)

                    except Exception as e:

                        st.error(
                            "Unable to generate the summary."
                        )

                        st.write(
                            f"Error: {e}"
                        )


        # -----------------------------
        # Display extracted information
        # -----------------------------
        st.header("3. Extracted document text")

        for page in pages:

            with st.expander(
                f"Page {page['page']}"
            ):

                if page["text"]:

                    st.write(
                        page["text"]
                    )

                else:

                    st.warning(
                        "No readable text found on this page."
                    )

        document.close()

    except Exception as e:

        st.error(
            "Unable to process this PDF."
        )

        st.write(
            f"Error: {e}"
        )


# -----------------------------
# Question section
# -----------------------------
st.divider()

st.header("4. Ask questions")

question = st.text_input(
    "Ask something about the uploaded document",
    placeholder=(
        "Example: What are the main applications discussed?"
    )
)


# -----------------------------
# Ask question
# -----------------------------
if st.button("Ask"):

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # Reconstruct document text
        document_text = "\n\n".join(
            f"Page {page['page']}:\n{page['text']}"
            for page in pages
        )

        if not document_text.strip():

            st.warning(
                "No readable text was found in this PDF."
            )

        else:

            with st.spinner(
                "Analyzing document..."
            ):

                try:

                    answer = generate_answer(
                        document_text,
                        question
                    )

                    st.subheader("Answer")

                    st.write(answer)

                except Exception as e:

                    st.error(
                        "Unable to generate an answer."
                    )

                    st.write(
                        f"Error: {e}"
                    )