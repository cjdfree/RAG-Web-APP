import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())


client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)


def gen_glm_params(prompt):
    '''
    构造 GLM 模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="glm-4", temperature=0.95):
    '''
    获取 GLM 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 glm-4，也可以按需选择 glm-3-turbo 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越稳定，温度系数越高，输出内容越随机。
    '''

    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"


# 调用API，输出对话结果
get_completion("你好")
# 这里对传入 zhipuai 的参数进行简单介绍：
# 共有top_p和temperature两种温度取样方式，建议您根据应用场景调整
# top_p (float)，用温度取样的另一种方法，称为核取样。取值范围是：(0.0, 1.0) 开区间，不能等于 0 或 1，默认值为 0.7。模型考虑具有 top_p 概率质量 tokens 的结果。例如：0.1 意味着模型解码器只考虑从前 10% 的概率的候选集中取 tokens
# request_id (string)，由用户端传参，需保证唯一性；用于区分每次请求的唯一标识，用户端不传时平台会默认生成

