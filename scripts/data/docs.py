from typing import List
import glob, os, re
import pymupdf4llm
import pandas as pd
from pprint import pprint
from langchain_community import document_loaders
from langchain_core.documents.base import Document



def get_pdf_documents(path:str, markdown_style=False)->List[Document]:
    """pdf 파일을 Document로 가지고 온다.

    Args:
        path (str): pdf 파일들이 들어가 있는 디렉터리의 경로 또는 pdf 파일의 경로
        markdown_style (bool, optional): pdf 파일을 markdown 형식으로 가지고 올지 여부(pdf 파일에 따라 안될 수 있음). Defaults to False.
            >>> markdown_style(pymupdf4llm)로 가지고 오는 경우, string 형식으로 출력된다.

    Returns:
        List[Document] | str: pdf로 부터 생성한 Document들이 들어 있는 List
    """
    if os.path.isdir(path):
        if len(glob.glob(f"{path}/*.pdf") + glob.glob(f"{path}/*.PDF")) > 0:
            docs = document_loaders.PyPDFDirectoryLoader(path=path).load()
    else:
        if markdown_style:
            docs = pymupdf4llm.to_markdown(path)
        else:
            docs = document_loaders.PyPDFLoader(file_path=path).load()
    return docs


# 전처리
class Cleaning:
    def __init__(self, docs:List[Document]):
        """
        Documents를 정규식 기반으로 전처리하는 class

        Args:
            docs (_type_): get_pdf_document를 통해 생성된 docs
        """
        self.docs = docs
        
    def __call__(self, fn, *args, **kwargs):
        for doc in self.docs:
            txt = doc.page_content
            doc.page_content = fn(txt, *args, **kwargs)
        
    # 공백을 모두 한 칸으로 수정
    def space_size(self, txt:str):
        return re.sub(r'\s+', ' ', txt)
    
    # remove_txt 제거
    def remove_specific_txt(self, txt:str, remove_txt:str):
        return re.sub(remove_txt, '', txt)
    
    # remove_txt를 공백으로 변경
    def specific_txt_to_space(self, txt:str, remove_txt:str):
        return re.sub(remove_txt, ' ', txt)
    
    # old 를 new로 변경
    def replace_specific_txt(self, txt:str, old:str, new:str):
        return re.sub(old, new, txt)
    
    # txt 앞 뒤 공백 제거
    def strip(self, txt:str):
        return txt.strip()


# 조회
class DocsFinder:
    def __init__(self, docs:List[Document]):
        """
        2개 이상의 pdf들이 하나로 합쳐진 Documents에 대하여, page와 파일의 일부 이름으로 그에 속하는 Documents만 가지고 온다.
        
        * 객체의 주 호출 메소드로. 이 메소드 하나로 클래스의 주요 기능이 구현됩니다. 인스턴스를 함수처럼 호출할 때 실행되며, 모든 주요 로직을
          여기에 구현 합니다.
        
        Example) instance = DocsFinder(documnts)
                 target_document = instance(page=42, file_contain_txt='guide_line_')
        
        Args:
            documents (List[Document]): 여러 pdf들로부터 생성된 Documents들이 하나로 합쳐진 상태
        """
        self.docs = docs
        self.metadata_df = self._make_metadata_df()
        
        
    def __call__(self, page:int=None, file_contain_txt:str=None):
        """
        파일 경로에 file_contain_txt를 포함하거나, page에 해당하는 Document만 조회
        >>> None에 대해선 Filter 되지 않음

        Args:
            page (int, optional): 찾고자 하는 page. Defaults to None.
            file_contain_txt (str, optional): 파일 경로에 포함되어 있는 명칭. Defaults to None.

        Returns:
            List[Document]: 조건에 해당하는 Documents
        """
        mask1 = self._get_mask1(txt=file_contain_txt)
        mask2 = self._get_mask2(page=page)
        return [self.docs[idx] for idx in set(mask1) & set(mask2)]
        
        
    def _make_metadata_df(self)->pd.core.frame.DataFrame:
        """document에 대한 metadata를 pd.DataFrame으로 가지고 온다.

        Returns:
            _type_: _description_
        """
        return pd.DataFrame([doc.metadata for doc in self.docs])
    
    
    # file_contain_txt 으로 mask 생성
    def _get_mask1(self, txt:str)->pd.core.indexes.base.Index | list:
        if txt is not None:
            mask1 = self.metadata_df[self.metadata_df['source'].str.contains(pat=txt)].index
        else:
            mask1 = self.metadata_df.index
        return mask1
    
    
    # page로 mask 생성
    def _get_mask2(self, page)->pd.core.indexes.base.Index | list:
        if page is not None:
            mask2 = self.metadata_df[self.metadata_df['page'] == page].index
        else:
            mask2 = self.metadata_df.index
        return mask2
    
    
# 시각화
class DocsViewer:
    def __init__(self, docs:List[Document]):
        """
        Documents를 보기 쉽게 하는 class
        
        * 객체의 주 호출 메소드로. 이 메소드 하나로 클래스의 주요 기능이 구현됩니다. 인스턴스를 함수처럼 호출할 때 실행되며, 모든 주요 로직을
          여기에 구현 합니다.
        
        Example) instance = DocsViewer(documnts)
                 instance(idx=100)

        Args:
            docs (List[Document]): langchain을 통해 만든 document
        """
        self.docs = docs
        self.len = len(docs)
        
        
    def __call__(self, idx:int, full_txt:bool=False):
        """_summary_

        Args:
            idx (int): docs의 index
            full_txt (bool, optional): 텍스트를 pprint()하지 않고, 문자로 출력. Defaults to False.
        """
        record = self.docs[idx]
        if full_txt:
            print(record.page_content)
        else:
            pprint(record.page_content)
            pprint(record.metadata)