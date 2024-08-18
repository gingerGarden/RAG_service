import os, warnings
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma, FAISS      # 로컬 Vector DB 사용
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.document_transformers import LongContextReorder
from transformers import AutoTokenizer




def split(
    docs:List[Document], 
    chunk_size:int, 
    chunk_overlap:int, 
    tokenizer:str=None,
    add_page_and_filename:bool=False
    )->List[Document]:
    """
    문서를 지정된 크기의 청크로 나눈다.
    >>> tokenizer가 지정되어 있다면, Token 수를 기준으로 문서를 청크 단위로 분할한다.

    Args:
        docs (List[Document]): 대상 Documents
        chunk_size (int): 문서를 나눌 청크의 크기
        chunk_overlap (int): 문서 간격 겹치는 크기
        tokenizer (str, optional): 청크를 토큰으로 나눌 Tokenizer, 만약 None이라면 적용하지 않는다.. Defaults to None.

    Returns:
        List[Document]: _description_
    """
    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, clean_up_tokenization_spaces=True)
        txt_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    else:
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return txt_splitter.split_documents(docs)



def add_filename_and_page_to_split_docs(split_docs:List[Document]):
    """split_docs에 파일의 이름과 페이지에 대한 내용을 상단에 추가한다.

    Args:
        docs (:List[Document]): 청크 단위로 쪼개진 Document
    """
    for doc in split_docs:
        filename = doc.metadata['source'].split("/")[-1]
        page = doc.metadata['page']
        doc.page_content = f"파일명: {filename}, 페이지: {page}, 내용: {doc.page_content}"



def get_vectorstore(
    embedding_model,
    split_docs=None, 
    save_path:str='./vectorstore', 
    makes_new:bool=False
    ):
    """
    문서 임배딩: 청크로 쪼개진 Documents를 embedding_model을 이용하여, 임배딩함.
    >>> 생성된 vectorstore는 save_path에 저장.
    >>> save_path에 이미 디렉터리가 존재하거나 makes_new=False인 경우, 그 save_path에서 가지고 온다.

    Args:
        embedding_model (_type_, optional): 임배딩 모델.
        split_docs (_type_): 청크 단위로 분할된 Documents.
        save_path (str): vectorstore를 저장할 디렉터리 경로. Defaults to './vectorstore'.
        makes_new (bool, optional): vectorstore를 새로 생성할 것인지 여부. Defaults to False.

    Returns:
        _type_: langchain_community.vectorstores.faiss.FAISS
    """
    # 생성이 되어 있지 않거나, makes_new=True인 경우, 새로 생성하고 저장
    if (os.path.exists(save_path) == False) | (makes_new == True):
        # vectorstore 생성
        vectorstore = FAISS.from_documents(
            documents=split_docs, embedding=embedding_model)
        # vectorstore 저장
        vectorstore.save_local(save_path)
    else:
        vectorstore = FAISS.load_local(
            save_path, embedding_model, allow_dangerous_deserialization=True
        )
    return vectorstore


class Retrievers:
    def __init__(self, 
            embedding_model, 
            faiss_path:str,
            split_docs:List[Document], 
            sparse_weight:float=0.5,
            k_size:int=4
            ):
        """기본적인 Retriever 3종류를 생성한다.
        1. 'dense': FAISS VectoreStore에서 유사도 기반으로 찾는다(의미 검색 중심).
        2. 'sparse': BM25 Retriever를 이용해 특정 키워드 포함으로 찾는다(키워드 중심).
        3. 'ensemble': BM25와 FAISS 두 가지 방법을 self.sparse_weight의 가중치로 찾는다.
        
        * 객체의 주 호출 메소드로. 주요 기능을 정리하여 사용 가능
        
        Example) instance = Retrievers(...)
                 retriever = instance(key='ensemble)

        Args:
            embedding_model (_type_): 임배딩하는 모델(OpenAI 또는 허깅페이스 모델)
            faiss_path (str): FAISS VectorStore가 저장되어 있는 디렉터리 경로
            split_docs (List[Document]): FAISS VectorStore 생성 전, 청크 단위로 Split된 Documents
            sparse_weight (float, optional): 앙상블 시, BM25 모델에 주는 가중치. Defaults to 0.5.
            k_size (int, optional): 가지고 오는 Document의 수(앙상블은 k_size/2의 내림을 하여 각각 실시). Defaults to 4.
        """
        self.faiss_vectorstore = get_vectorstore(embedding_model=embedding_model, save_path=faiss_path)
        self.split_docs = split_docs
        self.sparse_weight = sparse_weight
        self.k_size = k_size
        
    
    def __call__(self, key='dense'):
        if key == 'dense':
            return self.dense()
        elif key == 'sparse':
            return self.ensemble()
        elif key == 'ensemble':
            return self.sparse()
        else:
            warnings.warn("입력된 값이 적절하지 않습니다. 기본값인 FAISS vector store Retriever를 출력합니다.")
            return self.dense()
        
        
    # dense retriever와 sparse retriever 앙상블
    def ensemble(self):
        # sparse retriever 정의
        sparse = self.sparse(for_ensemble=True)
        # dense retriever 정의
        dense = self.dense(for_ensemble=True)
        # ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense, sparse], weights=[(1-self.sparse_weight), self.sparse_weight]
        )
        return ensemble_retriever
            
        
    # sparse retriever
    def sparse(self, for_ensemble:bool=False):
        # bm25 retriever 정의
        bm25_retiever = BM25Retriever.from_documents(self.split_docs)
        bm25_retiever.k = self.ensemble_size(for_ensemble=for_ensemble)
        return bm25_retiever
    
    
    # dense retriever
    def dense(self, for_ensemble:bool=False):
        inner_k_size = self.ensemble_size(for_ensemble=for_ensemble)
        faiss_retriever = self.faiss_vectorstore.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={"k": inner_k_size, "score_threshold": 0.5})
        return faiss_retriever
        
        
    # ensemble retriever 사용 시, 각 retirever에서 가져오는 Document의 크기를 고려해 k_size 조정
    def ensemble_size(self, for_ensemble:bool):
        inner_size = int(self.k_size//2) if for_ensemble else self.k_size
        if inner_size < 0:
            inner_size = 1
        return inner_size


# retriever가 찾아낸 문장들을 하나로 합친다.
def join_retrieved_docs(docs:List[Document], join_key:str="\n\n"):
    """document들을 join_key로 하나로 합침
    """
    return join_key.join(doc.page_content for doc in docs)


# Retriever가 찾아낸 Docs의 중요도에 따라 순서를 조정하여 문장을 하나로 합친다.
def raw_retrieved_docs_handler(retrieved_docs):
    # Retriever한 Docs들의 중요도에 따른 순서 조정
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(retrieved_docs)
    # Document들을 하나로 합친다.
    return join_retrieved_docs(reordered_docs)