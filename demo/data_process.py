import re
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档
# 读取PDF
loader = PyMuPDFLoader("data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")  # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
pdf_pages = loader.load()  # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")  # 类型list，长度

# list中每一个元素都是一个单独的文档，内容是单独的一页
# for pdf_page in pdf_pages:
#     print(f"每一个元素的类型：{type(pdf_page)}.", 
#         f"该文档的描述性数据：{pdf_page.metadata}", 
#         f"查看该文档的内容:\n{pdf_page.page_content}", 
#         sep="\n------\n")

# 读取md
loader = UnstructuredMarkdownLoader("data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
md_pages = loader.load()
print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")

# for md_page in md_pages:
#     print(f"每一个元素的类型：{type(md_page)}.", 
#         f"该文档的描述性数据：{md_page.metadata}", 
#         f"查看该文档的内容:\n{md_page.page_content}", 
#         sep="\n------\n")
    
# 2. 数据清洗（这一步往往很繁琐，显著影响后续知识库，RAG的质量）
# 我们期望知识库的数据尽量是有序的、优质的、精简的，因此我们要删除低质量的、甚至影响理解的文本数据。一般删除无意义的符号，空格，换行等

# 正则表达式清除换行\n
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_pages[0].page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
print(pdf_pages[0].page_content)

# 清楚'·'和空格
pdf_pages[0].page_content = pdf_pages[0].page_content.replace('•', '')
pdf_pages[0].page_content = pdf_pages[0].page_content.replace(' ', '')
print(pdf_pages[0].page_content)

# md两个换行符中间多了一个
md_pages[0].page_content = md_pages[0].page_content.replace('\n\n', '\n')
print(pdf_pages[0].page_content)


# 3. 文档分割
''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''

CHUNK_SIZE = 500  # 知识库中单段文本字符或 Token （如单词、句子等）的数量
OVERLAP_SIZE = 50  # 知识库中相邻文本重合长度，用于保持上下文的连贯性，避免分割丢失上下文信息

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
split_text = text_splitter.split_text(pdf_pages[0].page_content[0:1000])
split_docs = text_splitter.split_documents(pdf_pages)
print(f"切分后的文本数量：{len(split_text)}")
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")