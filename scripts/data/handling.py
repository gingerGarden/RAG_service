from typing import List
from langchain_core.documents.base import Document
from scripts.data.docs import Cleaning




def basic_cleaning(docs:List[Document]):
    """Documents에 대하여 공통적으로 적용하는 전처리

    Args:
        docs (List[Document]): Document가 들어있는 List
    """
    cleaning = Cleaning(docs=docs)
    # 공통적으로 적용되는 정규식
    cleaning(fn=cleaning.replace_specific_txt, old='\n', new=' ')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.replace_specific_txt, old=' ○ ', new=' ')
    cleaning(fn=cleaning.replace_specific_txt, old=' ~ ', new='-')
    cleaning(fn=cleaning.replace_specific_txt, old='~', new='-')
    cleaning(fn=cleaning.replace_specific_txt, old='「', new="'")
    cleaning(fn=cleaning.replace_specific_txt, old='」', new="'")
    cleaning(fn=cleaning.replace_specific_txt, old='\s?•\s?', new=". ")
    cleaning(fn=cleaning.replace_specific_txt, old=r'-+', new="-")
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='※')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='\uf000')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='□')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='Ÿ')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='ㅇ')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.strip)
    

def api_docs_cleaning(docs:List[Document]):
    """API Documents에 대하여 적용하는 전처리

    Args:
        docs (List[Document]): Document가 들어있는 List
    """
    # 공통 적용 정규식
    basic_cleaning(docs)
    # api 파일에 맞는 추가 정규식
    cleaning = Cleaning(docs=docs)
    cleaning(fn=cleaning.replace_specific_txt, old=r'·+', new=' ')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='-\s+\d+\s+-')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='금융분야 마이데이터 표준API 규격 ')
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='금융보안원 www.fsec.or.kr ')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.strip)
    
    
def guide_docs_cleaning(docs:List[Document]):
    """가이드 Documents에 대하여 적용하는 전처리

    Args:
        docs (List[Document]): Document가 들어있는 List
    """
    # 공통 적용 정규식
    basic_cleaning(docs)
    # 가이드 파일에 맞는 추가 정규식
    cleaning = Cleaning(docs=docs)
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt=r'Financial Security Institute \d+')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt='금융분야 마이데이터 기술 가이드라인 ')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.strip)
    cleaning(fn=cleaning.specific_txt_to_space, remove_txt=r'^\d+ 금융보안원')
    cleaning(fn=cleaning.space_size)
    cleaning(fn=cleaning.strip)