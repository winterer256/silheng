import streamlit as st
import os
import csv
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
#from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="실행론 AI 챗봇", page_icon="📄")
st.title("📄 실행론 AI 챗봇")
st.caption("sroberta + llama3.1로 구동되며, 대화 내용은 CSV로 자동 저장됩니다.")

# 2. RAG 체인 불러오기 (캐싱하여 속도 최적화)
@st.cache_resource
def load_rag_chain():
    # 원래 쓰시던 모델 그대로 복구!
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", 
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    db_data = vectorstore.get()
    docs = [Document(page_content=t, metadata=m) for t, m in zip(db_data['documents'], db_data['metadatas'])]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # ==========================================
    # 🚨 여기에 발급받은 Google API 키를 붙여넣으세요! 
    #os.environ["GOOGLE_API_KEY"] = "AIzaSyAkPGdbAuV68v6PAysUXe648vN97B3w-HA"
    # ==========================================

    # 구글 Gemini API 모델
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-latest", temperature=0)

    # Llama 3.1 로컬 모델
    #llm = ChatOllama(model="llama3.1", temperature=0) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 제공된 문서(Context)를 바탕으로 질문에 답하는 전문가입니다. \n"
                   "문서에 없는 내용은 지어내지 말고, 한국어로 자연스럽게 답변하세요.\n\n"
                   "[문서 내용]\n{context}"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(ensemble_retriever, combine_docs_chain)

rag_chain = load_rag_chain()

# 3. CSV 저장용 함수 정의
def log_to_csv(query, answer, contexts):
    log_file = "chat_history_log.csv"
    file_exists = os.path.isfile(log_file)
    source_info = " | ".join([str(doc.metadata) for doc in contexts])
    
    with open(log_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["시간", "질문", "AI 답변", "참고한 출처"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, query, answer, source_info])

# 4. 대화 기록 저장소 (Session State) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 이전 대화 기록 화면에 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        #메시지 출처 데이터가 숨어있다면 버튼 생성
        if "sources" in message:
            with st.expander("🔍 참고한 원문 확인하기"):
                st.markdown(message["sources"])

# 6. 하단 채팅 입력창 및 실행 로직
if prompt := st.chat_input("실행론 내용에 대해 무엇이든 물어보세요!"):
    
    # 6-1. 사용자 질문 출력
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 6-2. AI 답변 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("문서를 뒤적이며 답변을 작성하고 있습니다..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            contexts = response["context"]
            
            # 먼저 메인 답변만 깔끔하게 화면에 출력합니다.
            st.markdown(answer)
            
            # 출처 텍스트 조립하기
            source_text = ""
            for i, doc in enumerate(contexts):
                clean_content = doc.page_content.replace('\n', ' ')
                source_text += f"**{i+1}번째 출처:** `{doc.metadata}`\n"
                source_text += f"> {clean_content}\n\n"
            
            # 접기/펴기 버튼(Expander)을 만들고, 그 안에 출처 텍스트를 넣습니다.
            with st.expander("🔍 참고한 문서 원문 확인하기"):
                st.markdown(source_text)
            
            # 백그라운드에서 조용히 CSV 파일에 기록 저장!
            log_to_csv(prompt, answer, contexts)
            
    # 6-3. AI 답변을 세션에 저장
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": source_text  # 👈 핵심: 출처를 따로 저장해 둡니다.
    })
