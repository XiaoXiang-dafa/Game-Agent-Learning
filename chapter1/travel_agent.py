import os
import re
import requests
from urllib.parse import quote
from datetime import datetime, date
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient

load_dotenv(find_dotenv())

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not all([API_KEY, BASE_URL, MODEL_ID, TAVILY_API_KEY]):
    raise ValueError("请在项目根目录的 .env文件中配置 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID, TAVILY_API_KEY")

# 初始化 Tavily 客户端（复用）
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Open-Meteo 天气代码 → 中文描述
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "小冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}

_today = date.today().strftime("%Y-%m-%d")

AGENT_SYSTEM_PROMPT = f"""
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- get_weather(city: str, date: str): 查询城市天气。date 为可选参数（格式 YYYY-MM-DD），不填则查实时天气，填写则查指定日期预报（限今天±10天内）。当前日期为 {_today}，请根据用户的"明天""后天""大后天"等描述推算出准确的 YYYY-MM-DD 日期。
- get_attraction(city: str, weather: str):  根据城市和天气搜索推荐的旅游景点。务必使用 get_weather 返回的真实天气结果作为 weather 参数，而不是用户请求中的原始描述。

#输出格式要求:
你的每次回复必须严格遵循以下格式,包含一对Thought和Action:

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一:
1. 调用工具: function_name(arg_name="arg_value")
2. 结束任务: Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行,不要换行
- 当收集到足够信息可以回答用户时,必须使用Action: Finish[最终答案] 格式结束

请开始吧!
"""

def get_weather(city: str, date: str = None) -> str:
    """
    查询城市天气。
    - 不指定 date：查询实时天气（wttr.in）
    - 指定 date（YYYY-MM-DD）：查询该日天气预报（Open-Meteo，限±10天内）
    """

    # --- 查实时天气 ---
    if not date:
        url = f"https://wttr.in/{quote(city)}?format=j1"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            current_condition = data['current_condition'][0]
            weather_desc = current_condition['weatherDesc'][0]['value']
            temp_c = current_condition['temp_C']

            return f"{city}当前天气:{weather_desc}, 气温{temp_c}摄氏度"

        except requests.exceptions.HTTPError:
            return f"错误:找不到城市'{city}'的天气数据, 请检查城市名称是否正确"
        except requests.exceptions.RequestException as e:
            return f"错误:查询天气遇到网络问题 - {e}"
        except (KeyError, IndexError) as e:
            return f"错误:解析天气数据失败, 可能是城市名称无效 - {e}"

    # --- 查指定日期预报 ---
    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.now().date()
        diff = abs((target - today).days)

        if diff > 10:
            return f"错误:只能查询今天±10天内的天气，'{date}'相差{diff}天，超出范围"

        # 今天直接用实时天气
        if diff == 0:
            return get_weather(city)

        # 1. 地理编码：城市名 → 坐标
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(city)}&count=1&language=zh"
        geo_resp = requests.get(geo_url)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"错误:找不到城市'{city}'，请尝试用英文名或拼音"

        loc = geo_data["results"][0]
        lat = loc["latitude"]
        lon = loc["longitude"]
        resolved_name = loc.get("name", city)

        # 2. 获取天气预报
        fc_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min"
            f"&timezone=Asia/Shanghai"
            f"&start_date={date}&end_date={date}"
        )
        fc_resp = requests.get(fc_url)
        fc_resp.raise_for_status()
        fc_data = fc_resp.json()

        daily = fc_data["daily"]
        code = daily["weather_code"][0]
        t_max = daily["temperature_2m_max"][0]
        t_min = daily["temperature_2m_min"][0]

        desc = WEATHER_CODE_MAP.get(code, f"未知天气(code={code})")

        return f"{resolved_name} {date} 天气预报：{desc}，最高温{t_max}°C，最低温{t_min}°C"

    except requests.exceptions.HTTPError:
        return f"错误:查询'{city}'的天气预报失败，日期'{date}'可能超出可用范围"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气预报遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败 - {e}"
    




    
def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气, 使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 构造精确查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily_client.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答, 则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉, 没有找到相关的旅游景点推荐。"

        return "根据搜索, 为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
    
# 将所有工具函数放入一个字典, 方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}






class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)


    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role':'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误 - {e}")
            return "错误:调用语言模型服务时出错。"
        
if __name__ == "__main__":
    # --- 1. 配置LLM客户端 ---
    llm = OpenAICompatibleClient(
            model=MODEL_ID,
            api_key=API_KEY,
            base_url=BASE_URL
        )
    # --- 2. 初始化 ---
    user_prompt = "你好,帮我查询一下后天龙岩的天气, 然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求:{user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    # --- 3. 运行主循环 ---
    for i in range(5): # 设置最大循环次数
        print(f"--- 循环第 {i+1} 次 ---\n")

        # 3.1 构建Prompt
        full_prompt = "\n".join(prompt_history)

        # 3.2 调用LLM进行思考
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        # 模型可能会输出多余的Thought-Action, 需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 3.3 解析并执行行动
        action_match = re.search(r"Action:(.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到Action字段。请确保你的回复严格遵循'Thought:... Action:...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            fianl_answer = re.match(r"Finish\[(.*)]", action_str).group(1)
            print(f"任务完成, 最终答案: {fianl_answer}")
            break

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误: 为定义的工具 '{tool_name}'"

        # 3.4 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
