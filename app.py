import streamlit as st
from ingest import ingest_url
from rag import answer_query

st.title("Structured RAG over Website (Ollama)")

url = st.text_input("Enter website URL")

if url:
    st.components.v1.iframe(url, height=400)

    if st.button("Process Website"):
        with st.spinner("Ingesting..."):
            ingest_url(url)
        st.success("Processed!")

query = st.text_input("Ask a question")

if query:
    response = answer_query(query)
    st.write(response)
