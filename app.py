import os
import streamlit as st
from PIL import Image
import pytesseract
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\d407s798\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

INDEX_DIR = "storage/faiss_index"

# SYSTEM_PROMPT = """You are an AI Teaching Assistant for this course.
# Answer ONLY using the provided context. If the answer is not in the context, say you don't know and suggest where to look (e.g., slide/module).
# Always cite sources as: (Source: <filename> p.<page>).
# Keep answers concise and student-friendly. Render math in LaTeX when appropriate.
# """

SYSTEM_PROMPT = """You are teaching assistant for the class. The students will ask you to give feedback to charts they upload.
Take a look at the image files they upload and give them feedback only on errors you see. If text is not readable, i.e. it shows as '...' tell them to fix it. If they have a axis label on categorical axis tell them to delete it.
If they use 'Measure Names' as a legend title tell them to change it to something informative. If they use axis labels and a legend for the same
value tell them it's redundant. If they have added exact values by the bars on a bar chart and still have numeric axis and grid tell them they need
to declutter their plots. If they have overlapping elements, tell them to fix it."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nReturn a clear answer with citations.")
])

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src} p.{page+1}" if isinstance(page, int) else src
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] ({tag}) {snippet[:800]}")
    return "\n\n".join(lines)

def format_citation(meta):
    src = meta.get("source", "unknown")
    page = meta.get("page", None)
    return f"{src} p.{page+1}" if isinstance(page, int) else src

def main():
    st.set_page_config(page_title="BSAN 720 - AI Teaching Assistant", page_icon="🎓")
    st.title("🎓 BSAN 720 - AI Teaching Assistant")

    # Upload section
    st.markdown("### 📸 Upload a slide or photo to ask about it")
    uploaded_file = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])

    image_text = ""
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        try:
            image = Image.open(uploaded_file)
            image_text = pytesseract.image_to_string(image)
            if image_text.strip():
                st.success("Text extracted from image:")
                st.code(image_text[:1000], language="text")
            else:
                st.warning("Could not extract any text from the image.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Load vector store and LLM
    if "OPENAI_API_KEY" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()

    vs = load_vectorstore()
    retriever_k = st.sidebar.slider("Top-K retrieval", 3, 10, 5)
    threshold = st.sidebar.slider("Min similarity (0-1)", 0.0, 1.0, 0.2)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    default_prompt = "Ask a question about the course material..."
    if image_text.strip():
        default_prompt = "Ask a question about the uploaded image..."

    user_q = st.chat_input(default_prompt)
    if not user_q:
        return

    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    with st.spinner("Thinking..."):
        # Use image text if available
        full_query = user_q
        if image_text.strip():
            full_query = f"The student uploaded an image that says:\n\n{image_text.strip()}\n\nNow answer this question:\n{user_q}"

        docs_scores = vs.similarity_search_with_score(full_query, k=retriever_k)
        filtered = [(d, s) for d, s in docs_scores if s <= (1 - threshold)]
        docs = [d for d, _ in (filtered or docs_scores)]

        if not docs:
            bot = "I couldn’t find this in the course materials. Please check the syllabus or slides index."
        else:
            context = format_context(docs)
            prompt = ANSWER_PROMPT.format_messages(question=full_query, context=context)
            resp = llm.invoke(prompt)
            bot = resp.content

            citations = ", ".join(sorted({format_citation(d.metadata) for d in docs}))
            bot += f"\n\n**Sources:** {citations}"

    with st.chat_message("assistant"):
        st.markdown(bot)
    st.session_state.history.append(("assistant", bot))

if __name__ == "__main__":
    main()