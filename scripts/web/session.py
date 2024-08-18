import streamlit as st
import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory
        
        
                
def make_script_session(script_key:str):
    if script_key not in st.session_state:
        st.session_state[script_key] = []
        
        
def make_history_session(history_key:str):
    if history_key not in st.session_state:
        st.session_state[history_key] = {}
        
        
def reset_session_storage(script_key:str, history_key:str, session_id:str):
    st.session_state[script_key] = []
    st.session_state[history_key][session_id] = ChatMessageHistory()
    st.rerun()              # 스크립트 중단
        
        
def remove_session_storage(script_key:str, history_key:str, session_id:str):
    del(st.session_state[script_key])
    del(st.session_state[history_key][session_id])
    st.rerun()
        
    
def print_appended_scripts(script_key:str):
    """
    session의 script_key에 대화 내용이 존재하는 경우, role과 함께 모두 출력한다.

    Args:
        script_key (str): script가 st.session_state 에서 갖는 key 값
    """
    # script 저장 공간이 정의되어 있거나, script가 있는 경우 누적된 scripts 출력
    if script_key in st.session_state and len(st.session_state[script_key]) > 0:
        for script in st.session_state[script_key]:
            # script의 role(User or Bot) 아이콘과 script 같이 출력
            st.chat_message(script.role).write(script.content)
            
            
class SessionDF:
    def __init__(self, session_dict, unique_key, history_key):
        self.session_dict = session_dict
        self.unique_key = unique_key
        self.history_key = history_key
        
        
    def run(self):
        # session_id에 대한 list 생성
        session_id_list = self._make_session_id_list()
        # 기본적인 DataFrame 생성
        df = self._make_basic_session_id_df(session_id_list=session_id_list)
        # script size 컬럼 추가
        if len(df) > 0:
            # scripts의 수 
            self._add_script_size(df=df)
            # 기억하고 있는 QA의 수
            self._add_memorized_size(df=df)
            # Session id로 input되는 값을 컬럼에 추가
            self._add_input_session_id(df=df)
            df = df[["session_id", "real_session_id", "scripts_n", "history_n"]]
        return df
        
            
    def _make_basic_session_id_df(self, session_id_list):
        return pd.DataFrame({"real_session_id":session_id_list})
            
            
    def _make_session_id_list(self):
        key_list = list(self.session_dict.keys())
        if self.unique_key in key_list:
            key_list.remove(self.unique_key)
        if self.history_key in key_list:
            key_list.remove(self.history_key)
        return key_list
    
    
    def _add_script_size(self, df):
        df['scripts_n'] = [len(self.session_dict[session_id]) for session_id in df['real_session_id'].values]
        
        
    def _add_memorized_size(self, df):
        stack = []
        for session_id in df['real_session_id'].values:
            try:
                score = len(self.session_dict[self.history_key][session_id].messages)/2
            except:
                score = 0
            stack.append(score)
        df['history_n'] = stack
        
        
    def _add_input_session_id(self, df):
        unique_key = f"_{self.session_dict[self.unique_key]}"
        df["session_id"] = df['real_session_id'].str.replace(pat=unique_key, repl="")
        
        
        
        
    
    
    

        