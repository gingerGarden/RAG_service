# 환경변수 할당
from dotenv import load_dotenv
load_dotenv()

from scripts.data.docs import get_pdf_documents, DocsFinder, DocsViewer
from scripts.data.vectorstore import split, get_vectorstore
from scripts.data.handling import api_docs_cleaning, guide_docs_cleaning
from scripts.utils import do_or_load, new_dir_maker
from scripts.global_var import (
    RAWDATA_PATH, DOCS_PATH, SPLIT_DOCS_PATH, 
    TOKENIZER, EMBEDDING_MODEL, VECTORSTOR_STORE
)


# ---- Vectorstore 생성 시, 주요 인자 ----
vectorstore_path = VECTORSTOR_STORE
chunk_size, chunk_overlap = [int(i) for i in vectorstore_path.split("_")[-2:]]
embedding_model = EMBEDDING_MODEL


if __name__ == '__main__':
    
    # 단계 1. Load: pdf 파일들을 가지고 온다.
    # documents = get_pdf_documents(path='rawdata')
    documents= do_or_load(savepath=DOCS_PATH, makes_new=False, fn=get_pdf_documents, path=RAWDATA_PATH)
    find_docs = DocsFinder(docs=documents)     # pdf 가 여럿 있는 경우, 원하는 docs를 쉽게 찾아줌
    viewer_docs = DocsViewer(docs=documents)        # documents를 보기 쉽게함

    # documents에 대한 기본적인 전처리
    api_docs_cleaning(docs=find_docs(file_contain_txt='API'))          # API pdf 파일에 대한 전처리
    guide_docs_cleaning(docs=find_docs(file_contain_txt='가이드'))     # 가이드 pdf 파일에 대한 전처리

    # 단계 2. Split: Token 기반 문서 분할
    split_docs = do_or_load(
        savepath=SPLIT_DOCS_PATH, makes_new=False, fn=split, 
        docs=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=TOKENIZER)
    find_s_docs = DocsFinder(docs=split_docs)

    # 단계 3. EMBED: 문서 임배딩
    vectorstore = get_vectorstore(
        embedding_model=embedding_model, 
        split_docs=split_docs, 
        save_path=vectorstore_path, 
        makes_new=False)