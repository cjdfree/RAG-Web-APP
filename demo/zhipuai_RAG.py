import sys
import os

from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuAILLM

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser  # 结构化输出
from langchain.memory import ConversationBufferMemory  # 历史记忆
from langchain.chains import ConversationalRetrievalChain  # 对话检索链

from dotenv import load_dotenv, find_dotenv

# 加载环境变量，创建LLM
_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipuai_api_key)


# 加载已经存好的向量数据库
embedding = ZhipuAIEmbeddings()  # 定义 Embeddings
persist_directory = 'data_base/vector_db/chroma'  # 向量数据库持久化路径

vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")  # 检查数据库
question = "什么是prompt engineering?"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")
for i, doc in enumerate(docs):
    print(f"检索到的第{i+1}个内容: \n {doc.page_content}", end="\n----------------\n")


# 构建检索问答链
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""
# 提示词模板
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"], template=template)

# 基于模板的检索链
"""
    llm：指定使用的 LLM
    指定 chain_type="map_reduce，也可以利用load_qa_chain()方法指定chain type
    chain_type_kwargs = {"prompt": PROMPT}: 自定义 prompt
    return_source_documents=True参数, 返回源文档
    也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
"""
qa_chain = RetrievalQA.from_chain_type(zhipuai_model,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

# 问题测试
question_1 = "什么是南瓜书？"
question_2 = "Prompt Engineering for Developer是谁写的？"

result_1 = qa_chain({"query": question_1})
print(f"大模型+知识库后回答 question_1 的结果：\n {result_1['result']}")

result_2 = qa_chain({"query": question_2})
print(f"大模型+知识库后回答 question_2 的结果：\n {result_2['result']}")

# 大模型自己的回答
print(zhipuai_model.invoke(f"请回答下列问题:{question_1}"))  
print(zhipuai_model.invoke(f"请回答下列问题:{question_2}"))


# 添加历史记忆，构成完整的对话检索链
"""
    对话检索链（ConversationalRetrievalChain）在检索 QA 链的基础上，增加了处理对话历史的能力。
    它的工作流程是:
    1. 将之前的对话与新问题合并生成一个完整的查询语句。
    2. 在向量数据库中搜索该查询的相关文档。
    3. 获取结果后,存储所有答案到对话记忆区。
    4. 用户可在 UI 中查看完整的对话流程。
"""
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

qa = ConversationalRetrievalChain.from_llm(
    zhipuai_model,
    retriever=vectordb.as_retriever(),
    memory=memory
)
question = "我可以学习到关于提示工程的知识吗？"
result = qa({"question": question})
print(result['answer'])

question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])