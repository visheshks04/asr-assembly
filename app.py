import streamlit as st
from transcribe import transcribe
import tempfile
from dotenv import load_dotenv, find_dotenv
import datetime

_ = load_dotenv(find_dotenv())

current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)

if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter


llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

st.title('Audio Transcription with Speaker Diarization')
st.write('Upload an audio file to get the transcription with speaker diarization and timestamps.')

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    transcription, text = transcribe(temp_file_path)
    st.write("### Transcription") 
    for item in transcription:
        st.write(f"{item['timestamp']} seconds | {item['speaker']}: {item['text']}")


    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    text = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(text, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")

    if transcription:
        st.header('Ask Questions about the Transcript')
        query = st.text_input('Enter your question:')
        if 'history' not in st.session_state:
            st.session_state.history = []

        if query:
            docs = document_search.similarity_search(query)
            result = chain.invoke(input={'input_documents': docs, 'question': query})

            st.session_state.history.append({'question': query, 'answer': result['output_text']})
            if st.session_state.history:
                for qa in st.session_state.history:
                    st.write(f"You: {qa['question']}")
                    st.write(f"Reply: {qa['answer']}")

