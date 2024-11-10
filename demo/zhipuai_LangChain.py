from zhipuai_llm import ZhipuAILLM
# from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息
zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key)
zhipuai_model("你好，请你自我介绍一下！")

# 构建prompt

# 这里我们要求模型对给定文本进行中文翻译
# prompt = """请你将由三个反引号分割的文本翻译成英文！\
# text: ```{text}```
# """

# text = "我带着比身体重的行李，\
# 游入尼罗河底，\
# 经过几道闪电 看到一堆光圈，\
# 不确定是不是这里。\
# "
# prompt.format(text=text)
# print(prompt.format(text=text))

# 创建ChatPromptTemplate
template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
messages = chat_prompt.format_messages(input_language="中文", output_language="英文", text=text)
print(messages)
output  = zhipuai_model.invoke(messages)  # 使用LLM来回答
print(output, type(output))

"""
    output parser：输出解析器,将语言模型的原始输出转换为可以在下游使用的格式。 
    OutputParsers 有几种主要类型，包括：
    - 将 LLM 文本转换为结构化信息（例如 JSON） 
    - 将 ChatMessage 转换为字符串 
    - 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串

    最后，我们将模型输出传递给output_parser，是一个BaseOutputParser，
    接受的输入为字符串或BaseMessage。 
"""

output_parser = StrOutputParser()  # 将大模型输入转换为字符串
print(output_parser.invoke(output), type(output_parser.invoke(output)))

# 构建完整的链式流程
# 使用LCEL(LangChain Expression Language): chain = prompt | model | output_parser
chain = chat_prompt | zhipuai_model | output_parser
print(chain.invoke({"input_language":"中文", "output_language":"英文", "text": text}))

text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
print(chain.invoke({"input_language":"英文", "output_language":"中文", "text": text}))

