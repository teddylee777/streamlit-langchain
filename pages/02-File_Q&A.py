import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain import hub
import settings
import pickle
import os

# API KEY 를 설정합니다.
if "api_key" not in st.session_state:
    config = settings.load_config()
    if "api_key" in config:
        st.session_state.api_key = settings.load_config()["api_key"]
    else:
        st.session_state.api_key = ""

st.title("나만의 Chatbot")
st.markdown(
    f"""API KEY
    `{st.session_state.api_key[:-15] + '***************'}`
    """
)

if "history" not in st.session_state:
    st.session_state.history = []

if "user" not in st.session_state:
    st.session_state.user = []

if "ai" not in st.session_state:
    st.session_state.ai = []

if "model" not in st.session_state:
    st.session_state.model = "gpt-4-turbo-preview"


def add_history(role, content):
    if role == "user":
        st.session_state.user.append(content)
    elif role == "ai":
        st.session_state.ai.append(content)


def save_chat_history(title):
    pickle.dump(
        st.session_state.history,
        open(os.path.join("./chat_history", f"{title}.pkl"), "wb"),
    )


def load_chat_history(filename):
    with open(os.path.join("./chat_history", f"{filename}.pkl"), "rb") as f:
        st.session_state.history = pickle.load(f)
        print(st.session_state.history)
        st.session_state.user.clear()
        st.session_state.ai.clear()
        for user, ai in st.session_state.history:
            add_history("user", user)
            add_history("ai", ai)


def load_chat_history_list():
    files = os.listdir("./chat_history")
    files = [f.split(".")[0] for f in files]
    return files


model_name = st.empty()
tab1, tab2 = st.tabs(["Chat", "Settings"])


class StreamCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.full_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_text += token
        self.container.markdown(self.full_text)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


model_input = tab2.selectbox("model", ["gpt-3.5-turbo", "gpt-4-turbo-preview"], index=1)

if model_input:
    st.session_state.model = model_input
    model_name.markdown(f"#### {model_input}")

prompt_preset = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "You must answer in Korean, but DO NOT translate any technical terms. "
    "Use markdown to format your answer. "
)
prompt_input = tab2.text_area("Prompt", value=prompt_preset)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def print_history():
    for i in range(len(st.session_state.ai)):
        tab1.chat_message("user").write(st.session_state["user"][i])
        tab1.chat_message("ai").write(st.session_state["ai"][i])


with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )
    clear_btn = st.button("대화내용 초기화", type="primary", use_container_width=True)
    save_title = st.text_input(
        "저장할 제목",
    )
    save_btn = st.button("대화내용 저장", use_container_width=True)

    if clear_btn:
        st.session_state.history.clear()
        st.session_state.user.clear()
        st.session_state.ai.clear()
        print_history()

    if save_btn and save_title:
        save_chat_history(save_title)

    selected_chat = st.selectbox(
        "대화내용 불러오기", load_chat_history_list(), index=None
    )
    load_btn = st.button("대화내용 불러오기", use_container_width=True)
    if load_btn and selected_chat:
        load_chat_history(selected_chat)

if file:
    retriever = embed_file(file)


def create_prompt(custom_input):
    prompt = PromptTemplate.from_template(
        """{custom_prompt}

Question: {question} 
Context: {context} 
Answer:"""
    ).partial(custom_prompt=custom_input)
    return prompt


print_history()

if user_input := st.chat_input():
    add_history("user", user_input)

    tab1.chat_message("user").write(user_input)
    with tab1.chat_message("assistant"):
        if file is not None:
            msg = st.empty()
            llm = ChatOpenAI(
                model=st.session_state.model,
                temperature=0.1,
                streaming=True,
                callbacks=[StreamCallback(msg)],
                api_key=st.session_state.api_key,
            )
            prompt = create_prompt(prompt_input)

            # 체인을 생성합니다.
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(
                user_input
            )  # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            st.session_state.history.append((user_input, answer))
            add_history("ai", answer)
        else:
            st.write("먼저, 파일을 업로드해주세요.")
            add_history("ai", "먼저, 파일을 업로드해주세요.")
