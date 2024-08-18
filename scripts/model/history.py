from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(store_dict:dict, session_id:str) -> BaseChatMessageHistory:
    """
    store_dict의 session_id에 과거 대화 내용 저장

    Args:
        store_dict (dict): 세션 내 history가 저장된 dictionary
        session_id (str): 세션 내 history를 표시해주는 key

    Returns:
        BaseChatMessageHistory: chat message가 저장된 공간
    """
    # session_key가 store 내에 없는 경우
    if session_id not in store_dict:
        # 새로운 ChatMessageHistory 객체를 store_dict에 저장
        store_dict[session_id] = ChatMessageHistory()
    return store_dict[session_id]


# session의 history에 저장된 llm input(question), output 삭제
def remove_memory(store_dict:dict, session_id:str, maximum_size:int=5):
    """
    store(dict)에 session_id를 key로 저장되어 있는 모델의 input, output의 갯수가 maximum_size*2를 초과하는 경우,
    앞의 input, output 한쌍을 제거한다.

    Args:
        store_dict (dict): input, output이 저장된 dictionary
        session_id (str): session을 구분 짓는 key
        maximum_size (int, optional): 최대 저장 input, output의 쌍. Defaults to 5.
    """
    if len(store_dict[session_id].messages) > maximum_size*2:
        store_dict[session_id].messages.remove(store_dict[session_id].messages[0])
        store_dict[session_id].messages.remove(store_dict[session_id].messages[1])