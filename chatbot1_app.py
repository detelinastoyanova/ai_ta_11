# app.py
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate


INDEX_DIR = "storage/faiss_index"

SYSTEM_PROMPT = """You are an AI Teaching Assistant for this course.
Answer ONLY using the provided context. If the answer is not in the context, say you don't know and suggest where to look (e.g., slide/module).
Always cite sources as: (Source: <filename> p.<page>).
Keep answers concise and student-friendly. Render math in LaTeX when appropriate.
"""

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
    st.set_page_config(page_title="BSAN 720 AI Teaching Assistant", page_icon="🎓")
    st.title("🎓 BSAN 720 AI Teaching Assistant")

    # Secrets (Streamlit Cloud: Settings → Secrets)
    # Required: OPENAI_API_KEY
    if "OPENAI_API_KEY" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()

    vs = load_vectorstore()
    retriever_k = st.sidebar.slider("Top-K retrieval", 3, 10, 5)
    threshold = st.sidebar.slider("Min similarity (0-1)", 0.0, 1.0, 0.2)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # or gpt-4o/gpt-5 if enabled

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Ask about slides, topics, formulas, or examples…")
    if not user_q:
        st.info("Tip: ask “Where is CAPM covered?” or “Show the steps for covered interest parity.”")
        return

    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    with st.spinner("Thinking…"):
        # Retrieve
        docs_scores = vs.similarity_search_with_score(user_q, k=retriever_k)
        # Filter by threshold
        filtered = [(d, s) for d, s in docs_scores if s <= (1 - threshold)]  # FAISS lower=closer; convert if needed
        docs = [d for d, _ in (filtered or docs_scores)]

        if not docs:
            bot = "I couldn’t find this in the course materials. Please check the syllabus or slides index."
        else:
            context = format_context(docs)
            prompt = ANSWER_PROMPT.format_messages(question=user_q, context=context)
            resp = llm.invoke(prompt)
            bot = resp.content

            # Append explicit citations block
            citations = ", ".join(sorted({format_citation(d.metadata) for d in docs}))
            bot += f"\n\n**Sources:** {citations}"

    with st.chat_message("assistant"):
        st.markdown(bot)
    st.session_state.history.append(("assistant", bot))

if __name__ == "__main__":
    main()
