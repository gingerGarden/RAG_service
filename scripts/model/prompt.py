from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder


# 가장 기본적인 RAG 템플릿
rag_template1 = """
당신은 금융분야 마이데이터 설명에 특화되어있습니다.

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
I you don't know the anser, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
"""
rag_prompt1 = ChatPromptTemplate.from_template(rag_template1)


# Retriever가 찾아낸 문장들을 question을 기반으로 요약 정리 - Retriver to LLM
retrieved_handling_template = """
당신은 정리에 특화된 비서입니다.
Context의 각 문장들에서 Question과 관련 있는 부분을 정리하십시오. 중복된 내용들은 제거하십시오. 최대 5줄로 정리하십시오.

Question: {question}
Context: {context}
"""
retrieved_handling_prompt = ChatPromptTemplate.from_template(retrieved_handling_template)


# history가 반영되는 템플릿 - 성능이 좋았음(사용)
history_rag_prompt2 = ChatPromptTemplate.from_messages(
    [
        # 전역 설정
        (
            "system",
            """
            You are an assistant for question-answering for {ability} tasks. Use the following pieces of retrieved context to answer the question.
            I you don't know the anser, just say that you don't know. Use three sentences maximum and keep the answer concise.
            
            Context: {context}
            """,
        ),
        # 대화 기록
        MessagesPlaceholder(variable_name="history"),
        # 새로운 변수 - 사용자 입력
        ("human", "{question}"),
    ]
)



# retrieved_handling_template = """

# 당신은 정리에 특화된 비서입니다.
# Context의 각 문장들은 별개의 문장입니다. Question과 관련된 내용은 수정하지 말고, 중복된 내용들을 제거하여 최대 5줄로 Context를 정리하십시오.

# You are an assistant specialized in summarization.
# Each sentence in the context is independent. Summarize the context into a maximum of 5 lines by removing any redundant content, while keeping the information relevant to the question unchanged.

# Question: {question}
# Context: {context}
# """



# # history가 반영되는 템플릿 - 성능이 그다지 좋지 않음(사용 X)
# history_rag_template1 = """
# 당신은 금융분야 마이데이터 설명에 특화되어 있습니다.
# 만약 답을 모른다면, 모른다고 대답하십시오.
# 최대 3개 문장으로 간결하게 대답하십시오.
# Context와 History를 참고하여 질문에 대답하시오.

# Context: {context}
# History: {history}
# 질문: {question}
# """
# retrievalQA_style_prompt = PromptTemplate(
#     input_variables=["context", "question", "history"], 
#     template=history_rag_template1
# )



# history_rag_prompt = ChatPromptTemplate.from_messages(
#     [
#         # 전역 설정
#         (
#             "system",
#             """
#             당신은 {ability} 에 능숙한 어시스턴트입니다.
#             만약 답을 모른다면, 모른다고 대답하십시오. 최대 3개 문장으로 간결하게 대답하십시오.
#             Context를 사용하여 질문에 대답하시오.
            
#             Context: {context}
#             """,
#         ),
#         # 대화 기록
#         MessagesPlaceholder(variable_name="history"),
#         # 새로운 변수 - 사용자 입력
#         ("human", "{question}"),
#     ]
# )
