import streamlit as st
from scripts.web.session import SessionDF, remove_session_storage
from main import _UNIQUE_KEY, _HISTORY



st.title("생성된 Session ID 목록")

# Session 내 Session ID 기준으로, 적재된 데이터의 양 출력
TEST_INS = SessionDF(session_dict=st.session_state, unique_key=_UNIQUE_KEY, history_key=_HISTORY)
session_id_df = TEST_INS.run()
st.dataframe(session_id_df, use_container_width=True)

# Session ID 제거 버튼
st.title('Remove Session ID')
remove_session_id = st.text_input("지우고 싶은 Session ID의 real_session_id 입력 후 제거한다.")
if st.button("Delete"):
    remove_session_storage(script_key=remove_session_id, history_key=_HISTORY, session_id=remove_session_id)
    
# session.py 에 구현된 기능 설명
with st.sidebar:
    # Side Bar 기능 설명
    with st.expander('기능 설명'):
        st.markdown(
            """
            - 생성된 Session ID 목록: Session ID(session_id) 별 주요 속성. Session ID의 실제 값은 real_session_id이다.
            - Remove Session ID: 삭제하고자 하는 session_id 입력
            - Delete: Remove Session ID에 설정된 대상 영구 제거
            """
            )