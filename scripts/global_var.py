from scripts.utils import new_dir_maker

# Source 데이터의 경로 생성
SOURCE_PATH = "./source"
new_dir_maker(dir_path=SOURCE_PATH, makes_new=False)

# Rawdata의 경로
RAWDATA_PATH = "rawdata"

# PDF 파일들을 합쳐서 만들어낸 Documents의 중간 저장 경로
DOCS_PATH = "./source/docs.pickle"
# 청크 단위로 쪼개진 Documents의 저장 경로
SPLIT_DOCS_PATH = "./source/split_docs_300_100.pickle"
# Vectorstore(FAISS)의 저장 경로 - 뒤의 숫자는 chunk_size, chunk_overlap의 크기를 의미
VECTORSTOR_STORE = 'source/vectorstore_300_100'

# SPLIT_DOCS_PATH = "./source/split_docs_500_100.pickle"
# VECTORSTOR_STORE = 'source/vectorstore_500_100'
# SPLIT_DOCS_PATH = "./source/split_docs_300_50.pickle"
# VECTORSTOR_STORE = 'source/vectorstore_300_50'

# 허깅페이스를 통해 사용하는 Tokenizer 모델의 이름
TOKENIZER = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# Embedding 모델 정의 - Openai를 기본으로 사용
from scripts.model.model import cache_embedding_model
EMBEDDING_MODEL = cache_embedding_model(
    model_name='text-embedding-ada-002',
    cache_path="./source/cache"
)
# huggingface_embedding_model1 = "BAAI/bge-multilingual-gemma2"
# huggingface_embedding_model2 = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# LLM 모델
GPT = "gpt-3.5-turbo"                                   # openai
LLAMA1 = "llama-3-Korean-Bllossom-8B-Q4_K_M:latest"     # ollama
LLAMA2 = "Llama-3-Open-Ko-8B-Q8_0:latest"               # ollama