from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from scripts.utils import new_dir_maker



    
def cache_embedding_model(
    model_name:str='text-embedding-ada-002', 
    cache_path:str='./cache', 
    make_new:bool=False
    )->CacheBackedEmbeddings:
    """Embedding 모델을 cache에 연동하여 가지고 온다.
    openai = ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']
    Args:
        model_name (str, optional): Embedding 하는 모델의 이름. Defaults to 'text-embedding-ada-002'.
        cache_path (str, optional): Cache 파일의 저장 디렉터리 경로. Defaults to './cache'.
        make_new (bool, optional): Cache 파일의 저장 디렉터리를 새로 만들지 여부. Defaults to False.

    Returns:
        _type_: _description_
    """
    embedding_model = choose_embedding_model(model_name=model_name)
    return cache_embedder(embedding_model, cache_path=cache_path, make_new=make_new)
    
    
    
def choose_embedding_model(model_name:str):
    """Embedding할 모델의 이름에 맞는 method로 embedding 모델을 가지고 온다."""
    openai_list = ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']
    if model_name in openai_list:
        embedding_model = OpenAIEmbeddings(model=model_name)
    else:
        embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name)
    return embedding_model



def cache_embedder(embedding_model, cache_path:str, make_new:bool=False):
    """cache embedder 정의 및 데이터가 저장될 디렉터리 생성"""
    new_dir_maker(dir_path=cache_path, makes_new=make_new)
    # store 정의
    store = LocalFileStore(cache_path)
    result = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, store, namespace=embedding_model.model
    )
    return result
    


def get_llm(model_name='gpt-3.5-turbo', temperature:float=0, *args, **kwargs):
    """
    Openai 또는 Ollama로 llm 모델을 가지고 온다.

    Args:
        model (str, optional): model 이름. Defaults to 'gpt-3.5-turbo'.
        is_openai (bool, optional): openai 모델인지 여부. Defaults to True.
        temperature (float, optional): 0에 가까울수록 일관적인 값을 생성하며, 1에 가까울수록 다양하고 예측하기 힘든 답을 함. Defaults to 0.

    Returns:
        _type_: llm 모델
    """
    if is_openai_llm(model_name=model_name):
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, *args, **kwargs)
    else:
        llm = ChatOllama(model=model_name, temperature=temperature, *args, **kwargs)
    return llm


def is_openai_llm(model_name:str)->bool:
    """model_name이 openai llm list에 존재하는 경우, True 반환

    Args:
        model (str): get_llm에 들어간 모델의 이름

    Returns:
        bool: openai 모델에 속하는지 여부
    """
    openai_llm_list = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-4-turbo']
    if model_name in openai_llm_list:
        return True
    else:
        return False
    
    
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        """
        BaseCallbackHandler를 상속받아 Streaming하게 함.
        >>> LLM 모델의 응답이 한번에 완성되지 않고, 토큰이 생성될때마다 Container로 전달되도록 함.

        Args:
            container (_type_): 출력창
            initial_text (str, optional): 최초 문자열. Defaults to "".
        """
        self.container = container
        self.text = initial_text


    def on_llm_new_token(self, token: str, **kwargs)->None:
        self.text += token
        self.container.markdown(self.text)