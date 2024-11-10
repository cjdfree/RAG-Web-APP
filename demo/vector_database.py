import os
import re
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from zhipuai_embedding import ZhipuAIEmbeddings # 自定义Zhipu embedding


# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)


# 1. 加载文件
# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text 载入后的变量类型为`langchain_core.documents.base.Document, 文档变量类型包含两个属性: page_content 包含该文档的内容; meta_data` 为文档相关的描述性数据。
texts = []
for loader in loaders: texts.extend(loader.load())
# for text in texts:
#     print(f"每一个元素的类型：{type(text)}.", 
#         f"该文档的描述性数据：{text.metadata}", 
#         f"查看该文档的内容:\n{text.page_content[0:]}", 
#         sep="\n------\n")

# 2. 数据处理 
# 数据清洗
# 正则表达式清除换行\n
for text in texts:
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    text.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), text.page_content)  # 清除换行
    text.page_content = text.page_content.replace('•', '')  # 清除'·'
    text.page_content = text.page_content.replace(' ', '')  # 清楚空格
    text.page_content = text.page_content.replace('\n\n', '\n')  # 两个换行符中间多了一个
    print(text.page_content)

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(texts)



# 3. 创建数据库
embedding = ZhipuAIEmbeddings()  # 定义 Embeddings
persist_directory = 'data_base/vector_db/chroma'  # 定义持久化路径
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist() # 长期保存数据库
print(f"向量库中存储的数量：{vectordb._collection.count()}") # 向量数据库的数量


# 4. 数据库检索
question="什么是大语言模型"
# 相似度检索：检索最相关的内容
sim_docs = vectordb.similarity_search(question,k=3)  # similarity_search：数据库返回严谨的按余弦相似度排序的结果
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

# MMR检索：增加内容丰富度，但准确度较低
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")