"""
RAG 检索 + 问答链
"""

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

load_dotenv(override=True)

FAISS_DIR = "faiss_db"
TOP_K = 2

SYSTEM_PROMPT = """你是一名专业的电力检修智能助手，具备丰富的电力设备检修、故障诊断和安全操作知识。

请根据以下检索到的参考资料回答用户的问题。回答时请注意：
1. 优先基于参考资料中的内容作答，保证专业准确
2. 如果参考资料中没有相关信息，请如实说明，不要编造
3. 涉及安全操作时，务必强调安全注意事项
4. 回答要条理清晰，必要时使用编号列表

参考资料：
{context}

问题：{question}

请给出专业、准确的回答："""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT,
)


def build_chain():
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("Please set SILICONFLOW_API_KEY in .env file")

    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1",
    )
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1",
        temperature=0.1,
        timeout=60,
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask(chain, question: str):
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = result.get("source_documents", [])
    return answer, sources
