import streamlit as st

import os

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuAILLM

from dotenv import load_dotenv, find_dotenv


# 加载环境变量，创建LLM
_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']


# 请求API，返回大模型的回答
def generate_response(input_text, llm):
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output

def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

# 带有历史记录的问答链
def get_chat_qa_chain(question:str,zhipuai_api_key:str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model = "glm-4", temperature = 0.5, api_key = zhipuai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

# 不带历史记录的问答链
def get_qa_chain(question:str,zhipuai_api_key:str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model = "glm-4", temperature = 0.5, api_key = zhipuai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 zhipuai_streamlit_app hello world!')
    

    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 如果切换了模式，重置对话历史和 LLM
    if 'last_selected_method' not in st.session_state or st.session_state.last_selected_method != selected_method:
        st.session_state.messages = []  # 重置历史消息
        st.session_state.llm = None      # 重置LLM实例
        st.session_state.qa_chain = None # 重置问答链
        st.session_state.last_selected_method = selected_method  # 记录当前选择的模式

    # 初始化LLM实例和问答链
        if selected_method == "None":
            st.session_state.llm = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=zhipuai_api_key)
        elif selected_method == "qa_chain":
            st.session_state.qa_chain = get_qa_chain
        elif selected_method == "chat_qa_chain":
            st.session_state.qa_chain = get_chat_qa_chain

    # 创建消息框容器
    with st.container():
        messages_box = st.container()
        messages_box.write("对话内容：")  # 添加标题或说明

    # messages = st.container(height=300)

    # 处理用户输入
    if prompt := st.chat_input("请输入您的问题:"):
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 根据选择的模式生成回答
        if selected_method == "None" and st.session_state.llm:
            answer = generate_response(prompt, st.session_state.llm)
        elif selected_method == "qa_chain" and st.session_state.qa_chain:
            answer = st.session_state.qa_chain(prompt, zhipuai_api_key)
        elif selected_method == "chat_qa_chain" and st.session_state.qa_chain:
            answer = st.session_state.qa_chain(prompt, zhipuai_api_key)

        # 保存回答到历史记录并展示
        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            role = "assistant" if message["role"] == "assistant" else "user"
            st.write(f"{role}: {message['text']}")


if __name__ == "__main__":
    main()
