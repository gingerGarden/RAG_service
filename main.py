from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_core.messages import ChatMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from scripts.data.vectorstore import Retrievers, join_retrieved_docs, raw_retrieved_docs_handler
from scripts.model.prompt import retrieved_handling_prompt as ret_prompt
from scripts.model.prompt import history_rag_prompt2 as hist_prompt
from scripts.model.model import get_llm, StreamHandler
from scripts.model.history import remove_memory, get_session_history
from scripts.web.session import make_script_session, make_history_session, reset_session_storage, print_appended_scripts
from scripts.utils import load_pickle, now_datetime_string, random_str_id
from scripts.global_var import (
    SPLIT_DOCS_PATH, VECTORSTOR_STORE,
    GPT, LLAMA1, LLAMA2, EMBEDDING_MODEL
)


# ---- 전역 변수 정의 ----
_UNIQUE_KEY = "user_id"
_USER = "user"
_BOT = "assistant"
_HISTORY = "history"
_PROMPT_QUESTION_KEY = "question" 
_ABILITY = "Financial MyData"
_WAITING = "금융 마이데이터에 대해 궁금한 것을 입력해주세요." 
embedding_model = EMBEDDING_MODEL
vectorstore_path = VECTORSTOR_STORE


# ---- StreamLit UI 설정 ----
st.set_page_config(page_title="RAG(MyDATA)")
st.title("RAG (Financial MyDATA)")


# 해당 이용자의 Session내 고유 key 생성
if _UNIQUE_KEY not in st.session_state:
    st.session_state[_UNIQUE_KEY] = f"{now_datetime_string()}_{random_str_id(size=(10,13))}"


# ---- Side Bar ----
with st.sidebar:
    # session id
    session_id = st.text_input("Session ID:", value="my_session_000")
    session_id = f"{session_id}_{st.session_state[_UNIQUE_KEY]}"
    if st.button('Session Clear'):
        reset_session_storage(script_key=session_id, history_key=_HISTORY, session_id=session_id)
    # llm 모델 정의
    llm_model = st.selectbox("LLM", (GPT, LLAMA1, LLAMA2), index=0)
    # Retriever 정의
    retriever_key = st.selectbox("Retriever", ('dense', 'ensemble'), index=0)
    # Retriever LLM
    retriever_llm_key = st.selectbox("Retrieve, Generate", (True, False), index=1)
    # Retriever가 참조하는 문장의 갯수
    k_size = st.number_input(label="Documents size(1-10)", min_value=1, max_value=10, value=4, step=1)
    # memorized 크기
    memorized_size = st.number_input(label="Maximum memorize(3-10)", min_value=3, max_value=10, value=5, step=1)
    
    # Side Bar 기능 설명
    with st.expander('기능 설명'):
        st.markdown(
            """
            - Session ID: 해당 Session ID로 고유한 값
            - Session Clear: 해당 Session의 상태를 초기화 함
            - LLM: 자연어 생성 모델 선택. (Default: gpt-3.5-turbo)
            - Retriever: Ensemble(Dense, Sparse) 방식과 Dense(FAISS) 방식 중 선택. (Default: Dense)
            - Retriever, Generate: Retriever가 검색한 문서들을 LLM이 정리한 후 입력할지 여부. (Default: False)
            - Documents size(1-10): Retriever가 검색해올 문서의 수. (Default: 4)
            - Maximum memorize(3-10): 해당 Session에서 모델이 기억할 최대 문답의 갯수. (Default: 5)
            """
            )



# ---- 세션 저장소 설정 ----
# 세션 ID에 대한
# Session id 생성
# 세션 저장소(_SCRIPT, _HISTORY)가 존재하지 않는 경우, 새로 생성
make_script_session(script_key=session_id)
make_history_session(history_key=_HISTORY)
# 누적된 대화 내용 출력
print_appended_scripts(script_key=session_id)



# ---- Retriever 정의: LLM 모델을 통해 검색 결과를 정리하는 chain ----
_ret_ins = Retrievers(
    embedding_model=embedding_model, 
    faiss_path=vectorstore_path,
    split_docs=load_pickle(SPLIT_DOCS_PATH),
    sparse_weight=0.2,
    k_size=k_size
    )
retriever = _ret_ins(key=retriever_key)
ret_llm = get_llm(model_name=llm_model, temperature=0)
ret_chain = (
    {"context": retriever | join_retrieved_docs, "question":RunnablePassthrough()}
    # {"context": retriever | raw_retrieved_docs_handler, "question":RunnablePassthrough()}
    | ret_prompt
    | ret_llm
    | StrOutputParser()
)



# ---- Scripts ----
if user_input := st.chat_input(_WAITING):
    # 사용자 입력
    st.chat_message(_USER).write(user_input)                                        # 입력 출력
    st.session_state[session_id].append(ChatMessage(role=_USER, content=user_input))   # 유저 입력 script 세션 저장
    # retriever_llm_key에 따라 LLM 모델을 이용하여, Retriever 내용을 정리할지 여부
    handled_context = ret_chain.invoke(user_input) if retriever_llm_key else join_retrieved_docs(retriever.invoke(user_input))
    
    # Bot 답변 출력 및 저장
    with st.chat_message(_BOT):
        
        # ---- chain 설정 ----
        streaming = StreamHandler(container=st.empty())     # llm으로부터 전달받는 값을 Token 단위로 출력(BOT script 출력)
        llm = get_llm(model_name=llm_model, streaming=True, callbacks=[streaming])      # llm 모델 설정
        chain = (
            hist_prompt
            | llm
            | StrOutputParser()
        )
        with_message_history = (
            RunnableWithMessageHistory(
                chain,
                get_session_history=lambda session_id: get_session_history(st.session_state[_HISTORY], session_id),
                input_messages_key=_PROMPT_QUESTION_KEY,
                history_messages_key=_HISTORY,
            )
        )
        # LLM을 이용해 답변 생성 및 저장
        bot_reply = with_message_history.invoke(
            {'ability':_ABILITY, 'question':user_input, 'context':handled_context}, # hist_prompt에 내용 전달
            config={"configurable":{"session_id":session_id}}                       # history 저장을 위한 session_id(key) 전달
        )
        # 메모리가 일정 수준 이상 쌓이면, 오래된 것부터 제거
        remove_memory(store_dict=st.session_state[_HISTORY], session_id=session_id, maximum_size=memorized_size)
        # BOT의 출력 문구 Session 내 저장
        st.session_state[session_id].append(ChatMessage(role=_BOT, content=bot_reply))