import pandas as pd
import numpy as np
import json
import re
import sys
import io
import copy
import multiprocessing
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import folium
import gurobipy as gp
from gurobipy import GRB
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
import subprocess
import tempfile
import os
warnings.filterwarnings('ignore')

# ==================== DeepSeek API 客户端 ====================
import requests
import json
import time

class DeepSeekAPI:  # 保持类名不变，以免改动后面的代码
    def __init__(self, api_key=None, provider="qwen"):
        self.provider = provider.lower()
        
        # 1. 阿里云 Qwen (国内直连，极速稳定)
        if self.provider == "qwen":
            # 移除硬编码，改用环境变量兜底
            self.api_key = api_key or os.getenv("QWEN_API_KEY")
            self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            self.model_name = "qwen-max"
            
        # 2. DeepSeek 官方 (国内直连)
        elif self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.api_url = "https://api.deepseek.com/chat/completions"
            self.model_name = "deepseek-chat"
            
        # 3. GPT-4o 中转代理 (国内直连，需替换中转平台的URL)
        elif self.provider == "gpt-5.2":
            self.api_key = api_key or os.getenv("GPT_PROXY_API_KEY")
            # 注意：这里千万别填 api.openai.com，填中转平台给你的国内可用域名
            self.api_url = "https://jbridge.ai/api/v1/chat/completions" 
            self.model_name = "gpt-5.2"
            
        else:
            raise ValueError(f"不支持的模型提供商: {self.provider}")

        # 核心安全校验：如果没有获取到 API Key，立即报错拦截
        if not self.api_key:
            raise ValueError(
                f"🚨 未检测到 {self.provider} 的 API 密钥！\n"
                f"请在运行前设置环境变量（例如 export {self.provider.upper()}_API_KEY='your_key'），"
                "或在交互界面中手动输入。"
            )
            

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
            

    def query(self, prompt: str, system_prompt: str = None, temperature: float = 0.1) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # ⚠️ 修复核心：这里去掉了 "max_tokens": 2000，这是 99% 代理报 400 的元凶！
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        for attempt in range(3):
            try:
                import time
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=90)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429: # 触发限流
                    print(f"⚠️ 触发限流，等待2秒后重试... ({attempt+1}/3)")
                    time.sleep(2)
                    continue
                else:
                    # 🚨 极其关键：把 JBridge 官方的报错真言打印出来！
                    print(f"\n🚨 [JBridge 官方详细报错]: {response.text}\n")
                    return f"API 错误: {response.status_code}"
            except Exception as e:
                print(f"⚠️ 网络请求异常，正在重试... ({attempt+1}/3) | 报错: {str(e)}")
                import time
                time.sleep(3)
                
        return "请求失败: 网络彻底断开或达到最大重试次数"

    def query_json(self, prompt: str, system_prompt: str = None) -> dict:
        response = self.query(prompt, system_prompt, temperature=0.1)
        try:
            return json.loads(response)
        except:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        return {}

# ==================== 问题类型枚举 ====================
class ProblemType(Enum):
    TIMETABLING = "timetabling"
    HEADWAY = "headway"
    CAPACITY = "capacity"
    DELAY = "delay"
    RESCHEDULING = "rescheduling"
    ROLLING_STOCK = "rolling_stock"
    CIRCULATION = "circulation"
    CREW = "crew"
    PASSENGER_ASSIGNMENT = "passenger_assignment"
    CROWDING = "crowding"
    ENERGY = "energy"
    REGENERATIVE = "regenerative"
    NETWORK_FLOW = "network_flow"
    INTEGRATED_MTR = "integrated_mtr"

# ==================== Query Intent ====================
class QueryIntent(Enum):
    OPTIMIZE = "optimize"
    EXPLAIN = "explain"
    ADAPT = "adapt"

# ==================== Intent Classifier (大模型智能语义版) ====================
class IntentClassifier:
    def __init__(self, api_client: DeepSeekAPI):
        self.api = api_client

    def classify(self, query: str) -> str:
        system_prompt = (
            "你是一个精通自然语言理解与城市轨道交通业务的意图识别专家。\n"
            "你的任务是根据用户的提问，判断用户的核心意图。\n"
            "必须严格输出为标准的 JSON 纯文本格式，包含一个键 'intent'，不要有任何多余文字。"
        )
        prompt = f"""
请分析以下用户的地铁调度相关查询：
"{query}"

可选的意图列表（请仔细区分业务场景）：
- "optimize": 求解与优化。用户希望系统计算出一个最优结果、生成调度策略、求解排班或延误问题（例如："求最少列车数"、"如何调整发车间隔"、"求最小化残留延误"、"发生延误如何恢复"等）。
- "explain": 解释与分析。用户希望了解上一次求解结果的原因、查看特定约束、或对已有方案进行解读（例如："为什么发车间隔是120秒"、"解释一下刚刚的方案"）。
- "adapt": 假设与敏感性分析。用户提出在“当前已有结果”的基础上修改某个宏观外在条件，看对整体有什么影响（例如："如果整体客流增加20%会怎样"、"假设突然坏了2辆车怎么办"）。

【意图识别防坑指南】：
- 只要题目中明确带有求解指令（如“求最优策略”、“求几阶延误”、“最小化XX”），即使句子中包含了“提高速度”、“压缩时间”、“减少”，也【绝对】属于 "optimize"，因为这些是数学模型里的控制策略，而不是敏感性分析。
- 绝大多数包含具体数字参数（如延误60分钟、8列车）的场景推演题都是 "optimize"。

请输出分类结果，示例：
{{"intent": "optimize"}}
"""
        response_dict = self.api.query_json(prompt, system_prompt=system_prompt)
        
        # 获取大模型的分类结果，如果发生异常则兜底返回 optimize
        predicted_intent = response_dict.get("intent", QueryIntent.OPTIMIZE.value)
        return predicted_intent.lower()

# ==================== Query Processor ====================
class QueryProcessor:
    def process(self, query: str) -> str:
        return query.strip()

# ==================== Problem Classifier (大模型智能语义版) ====================
class ProblemClassifier:
    def __init__(self, api_client: DeepSeekAPI):
        self.api = api_client

    def classify(self, query: str) -> str:
        system_prompt = (
            "你是一个精通城市轨道交通运筹优化建模的意图识别专家。\n"
            "你的任务是根据用户的提问，将其归类到最匹配的数学规划问题类型。\n"
            "必须严格输出为标准的 JSON 纯文本格式，包含一个键 'type'，不要有任何多余文字。"
        )
        prompt = f"""
请分析以下地铁调度问题：
"{query}"

可选的问题类型列表（请仔细区分业务场景）：
- "timetabling": 时刻表优化
- "headway": 常规发车间隔优化（以乘客等待时间最小化为主要目标）
- "capacity": 运力或班次配置（纯粹为了满足客流和负载率，最少需要多少资源或班次）
- "delay": 延误传播分析与恢复（题目提及初始延误、恢复系数，目的是消除延误）
- "rescheduling": 故障重调度（车次取消、运行图打乱后的重新排班）
- "rolling_stock": 车底分配与调度
- "circulation": 交路规划
- "crew": 乘务员/司机排班
- "energy": 节能与单车牵引能耗优化（求解需要距离和限速）。
- "regenerative": 再生制动能量利用与多车同步优化（【防坑指南】：只要题目中出现“再生能量”、“制动”、“对向牵引”、“利用率”、“发车时距偏移”，“牵引负载”等词，归类于此，不可归为 energy！）
- "passenger_assignment": 客流分配、乘客路径选择、网络流量分配（【防坑指南】：当题目出现“节点”、“OD”、“求分配”字眼时，即使带有具体客流数字，归类于此，不可归为 capacity！）
- "crowding": 站台拥挤与排队论优化（【防坑指南】：只要题目出现“到达率”、“疏散能力”、“排队”、“滞留人数”等词，归类于此，不可归为 capacity！）

【意图识别防坑指南】：
- 如果是在“延误/晚点”场景下求发车间隔，必须归类为 "delay"，而不是 "headway"。
- 如果是求“最经济班次”、“最小配置运力”，即使提到了发车间隔，也必须归类为 "capacity"。

请输出分类结果，示例：
{{"type": "delay"}}
"""
        response_dict = self.api.query_json(prompt, system_prompt=system_prompt)
        
        # 获取大模型的分类结果，如果发生异常则兜底返回 headway
        predicted_type = response_dict.get("type", ProblemType.HEADWAY.value)
        return predicted_type.lower()

# ==================== 智能参数提取器（基于大模型） ====================
class ParameterExtractor:
    def __init__(self, api_client: DeepSeekAPI):
        self.api = api_client

    def extract(self, query: str, problem_type: str, required_params: List[str]) -> Dict[str, Any]:
        system_prompt = (
            "你是一个精通城市轨道交通（MTR）运筹优化建模的参数提取专家。\n"
            "你的任务是从用户输入的问题中，准确提取出数学规划模型所需的关键结构化参数。\n"
            "必须严格输出为标准的 JSON 纯文本格式，不要包含任何 Markdown 格式标记（如 ```json），不要包含任何解释性文字。"
        )
        prompt = f"""
用户求解的问题: "{query}"
识别出的问题类型: "{problem_type}"
该模型核心需要的参数列表: {required_params}

请根据用户文本提取这些参数的值。业务逻辑与换算规则提示：
1. 提取的值必须是符合实际的数字（整型/浮点型）或字符串。例如："40分钟" -> 40, "5列" -> 5, "金钟站" -> "金钟"。
2. 如果文本中提及了具体的地铁线路名称（如 荃湾线、港岛线、东涌线、东铁线），请务必统一提取到 "line" 键中（即便它不在核心参数列表中）。
3. 如果文本中完全没有提及某核心参数，请不要将其放入 JSON 中，或者将其显式设为 null。
4. 确保输出的 JSON 键（Key）与要求的参数名完全一致。

期望输出的 JSON 格式示例：
{{
    "line": "荃湾线",
    "delay_minutes": 40,
    "affected_section": "金钟"
}}
"""
        response_text = self.api.query(prompt, system_prompt=system_prompt, temperature=0.0)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        try:
            extracted_params = json.loads(clean_text)
            final_params = {k: v for k, v in extracted_params.items() if v is not None}
            return final_params
        except Exception as e:
            print(f"⚠️ [Extractor] 大模型参数 JSON 解析失败: {e}. 原始文本: {clean_text}")
            return {}

# ==================== Parameter Requirement Library (补丁) ====================
class ParameterRequirementLibrary:
    REQUIREMENTS = {
        "headway": ["Q", "C", "N"],
        "capacity": ["Q", "C"],
        "delay": ["delay_minutes"],
        "rescheduling": ["delay_minutes"],
        "rolling_stock": ["trip_set", "fleet_size"],
        "circulation": ["trip_set"],
        "crew": ["trip_set", "crew_size"],
        # 修改：将 "OD_matrix" 替换为 "OD_matrix_total"，适应宏观总量场景
        "passenger_assignment": ["OD_matrix_total"],
        # 修改：明确排队论关键参数
        "crowding": ["station_flow_arrival", "max_platform_capacity"],
        "energy": ["distance", "speed_limit"],
        "regenerative": ["generated_power", "required_power"]
    }

    @classmethod
    def get_required_params(cls, problem_type: str) -> List[str]:
        return cls.REQUIREMENTS.get(problem_type, [])

# ==================== Infrastructure KB ====================
class InfrastructureKB:
    LINES = {
        "东铁线": {
            "run_time": 50,
            "turnback": 5,
            "cycle_time": 110,
            "min_headway": 120,
            "max_headway": 600,
            "capacity": 1500
        },
        "荃湾线": {
            "run_time": 35,
            "turnback": 5,
            "cycle_time": 80,
            "min_headway": 120,
            "max_headway": 480,
            "capacity": 1500
        },
        "港岛线": {
            "run_time": 28,
            "turnback": 4,
            "cycle_time": 64,
            "min_headway": 120,
            "max_headway": 480,
            "capacity": 1500
        },
        "东涌线": {
            "run_time": 45,
            "turnback": 6,
            "cycle_time": 102,
            "min_headway": 150,
            "max_headway": 600,
            "capacity": 1500
        }
    }

    @classmethod
    def get_line(cls, line_name):
        return cls.LINES.get(line_name)

# ==================== Operation Policy KB ====================
class PolicyKB:
    POLICY = {
        "target_load_factor": 0.85,
        "recovery_factor": 0.6,
        "max_short_turn_ratio": 0.30,
        "delay_weight": 0.80,
        "energy_weight": 0.20
    }

    @classmethod
    def get(cls, key, default=None):
        return cls.POLICY.get(key, default)

# ==================== Model Adaptation Agent (降敏版) ====================
class ModelAdaptationAgent:
    def __init__(self):
        # 扩展受控适应的允许范围
        self.allowed_adaptations = {
            "headway": ["fleet_constraint", "delay_constraint", "capacity_constraint"],
            "timetabling": ["delay_constraint", "energy_constraint"],
            "rescheduling": ["delay_constraint", "fleet_constraint"],
            "capacity": ["target_load_factor_constraint"],
            "crowding": ["platform_capacity_constraint"]
        }

    def adapt(self, problem_type: str, query: str, template: Dict) -> Dict:
        adaptation_trace = []
        adapted_model = copy.deepcopy(template)
        allowed = self.allowed_adaptations.get(problem_type, [])
        
        # 1. 车底/资源受限场景激活 (避开普通的"车底"、"固定"，改用极端词汇)
        if any(kw in query for kw in ["锁定车底", "仅有", "最多可用", "车辆短缺", "应急抽调"]) and "fleet_constraint" in allowed:
            adapted_model.setdefault("constraints", []).append("fleet_limit")
            adaptation_trace.append({
                "trigger": "检测到车辆资源极端受限", 
                "adaptation": "注入 fleet_constraint", 
                "reason": "防范过度调度，动态激活车辆总数硬约束上限。"
            })
            
        # 2. 延误传播场景激活
        if any(kw in query for kw in ["延误", "晚点", "故障", "恢复"]) and "delay_constraint" in allowed:
            adapted_model.setdefault("constraints", []).append("delay_recursion")
            adaptation_trace.append({
                "trigger": "检测到故障/延误场景", 
                "adaptation": "注入 delay_constraint", 
                "reason": "自动引入延误衰减与传播方程，将静态模型转化为动态推演模型。"
            })
            
        # 3. 拥挤/负载率红线激活 (避开普通的"安全限制"，改用业务预警词汇)
        if any(kw in query for kw in ["拥挤风险", "站台爆满", "大客流积压", "超载红线"]) and "capacity_constraint" in allowed:
            adapted_model.setdefault("constraints", []).append("load_factor")
            adaptation_trace.append({
                "trigger": "检测到客流拥挤预警", 
                "adaptation": "注入 capacity_constraint", 
                "reason": "强制设定负载率红线，以降低系统效率为代价换取绝对安全裕度。"
            })
            
        return {
            "base_model": template.get("name"), 
            "adapted_model": adapted_model, 
            "adaptation_trace": adaptation_trace
        }

# ==================== 香港地铁数据加载器 ====================
class HKMetroDataLoader:
    @staticmethod
    def load_stations(stations_text: str) -> Dict[str, Dict]:
        stations = {}
        lines = stations_text.strip().split('\n')
        for i, line in enumerate(lines):
            if i == 0: continue
            parts = line.strip().split(',')
            if len(parts) >= 5:
                station_id = parts[0]
                stations[station_id] = {
                    'station_id': station_id,
                    'name_cn': parts[1],
                    'name_en': parts[2],
                    'longitude': float(parts[3]),
                    'latitude': float(parts[4]),
                    'platform_count': int(parts[5]) if len(parts) > 5 else 2,
                    'lines': [],
                    'passenger_flow': {}
                }
        return stations

    @staticmethod
    def load_lines(lines_text: str) -> Dict[str, Dict]:
        lines = {}
        data_lines = lines_text.strip().split('\n')
        for i, line in enumerate(data_lines):
            if i == 0: continue
            parts = line.strip().split(',')
            if len(parts) >= 5:
                line_id = parts[0]
                lines[line_id] = {
                    'line_id': line_id,
                    'name_cn': parts[1],
                    'name_en': parts[2],
                    'color': parts[3],
                    'train_type': parts[4],
                    'stations': [],
                    'segments': []
                }
        return lines

    @staticmethod
    def load_sections(sections_text: str) -> List[Dict]:
        segments = []
        lines = sections_text.strip().split('\n')
        for i, line in enumerate(lines):
            if i == 0: continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                segment_id = parts[0]
                segments.append({
                    'segment_id': segment_id,
                    'from_station_id': parts[1],
                    'to_station_id': parts[2],
                    'distance_km': float(parts[3]),
                    'runtime_sec': int(parts[4]),
                    'stop_time_sec': int(parts[5]),
                    'line_id': parts[6],
                    'is_connected': parts[7].lower() == 'true' if len(parts) > 7 else True
                })
        return segments

    @staticmethod
    def create_network() -> Dict[str, Any]:
        print("🚇 构建香港地铁网络...")
        stations_text = """station_id,name_cn,name_en,longitude,latitude,platform_count
STN001,金钟,Admiralty,114.167700,22.278200,4
STN002,中环,Central,114.159500,22.280800,5
STN003,尖沙咀,Tsim Sha Tsui,114.171700,22.293300,2
STN004,旺角,Mong Kok,114.170800,22.319700,2
STN005,太子,Prince Edward,114.167500,22.323800,2
STN006,深水埗,Sham Shui Po,114.162500,22.332200,2
STN007,长沙湾,Cheung Sha Wan,114.157500,22.335500,2
STN008,荔枝角,Lai Chi Kok,114.150000,22.338000,2
STN009,美孚,Mei Foo,114.141700,22.338300,4
STN010,荃湾,Tsuen Wan,114.112200,22.362500,2
STN011,北角,North Point,114.192500,22.285500,2
STN012,鲗鱼涌,Quarry Bay,114.204200,22.284200,2
STN013,西湾河,Sai Wan Ho,114.215000,22.281700,2
STN014,柴湾,Chai Wan,114.241700,22.264700,2
STN015,香港,Hong Kong,114.158300,22.289700,2
STN016,九龙,Kowloon,114.168300,22.302500,2
STN017,青衣,Tsing Yi,114.108300,22.321700,3
STN018,东涌,Tung Chung,113.932500,22.291700,2
STN019,欣澳,Sunny Bay,114.028300,22.318300,2
STN020,迪士尼,Disneyland,114.048300,22.313300,1"""
        lines_text = """line_id,name_cn,name_en,color,train_type
LINE01,荃湾线,Tsuen Wan Line,红色,MTR Metro Cammell
LINE02,港岛线,Island Line,蓝色,MTR Metro Cammell
LINE03,东涌线,Tung Chung Line,橙色,Adtranz–CAF
LINE04,迪士尼线,Disneyland Resort Line,粉红色,Adtranz–CAF
LINE05,机场快线,Airport Express,绿色,CAF Trains"""
        sections_text = """section_id,from_station_id,to_station_id,distance_km,runtime_sec,stop_time_sec,line_id,is_connected
SEC001,STN002,STN003,1.4,120,20,LINE01,True
SEC002,STN003,STN002,1.4,120,20,LINE01,True
SEC003,STN003,STN004,1.6,130,20,LINE01,True
SEC004,STN004,STN003,1.6,130,20,LINE01,True
SEC005,STN004,STN005,0.8,70,20,LINE01,True
SEC006,STN005,STN004,0.8,70,20,LINE01,True
SEC007,STN005,STN006,0.9,75,20,LINE01,True
SEC008,STN006,STN005,0.9,75,20,LINE01,True
SEC009,STN006,STN007,0.9,80,20,LINE01,True
SEC010,STN007,STN006,0.9,80,20,LINE01,True
SEC011,STN007,STN008,1.0,85,20,LINE01,True
SEC012,STN008,STN007,1.0,85,20,LINE01,True
SEC013,STN008,STN009,1.1,90,20,LINE01,True
SEC014,STN009,STN008,1.1,90,20,LINE01,True
SEC015,STN009,STN010,4.2,240,20,LINE01,True
SEC016,STN010,STN009,4.2,240,20,LINE01,True
SEC017,STN002,STN001,0.7,60,20,LINE02,True
SEC018,STN001,STN002,0.7,60,20,LINE02,True
SEC019,STN001,STN011,2.2,150,20,LINE02,True
SEC020,STN011,STN001,2.2,150,20,LINE02,True
SEC021,STN011,STN012,1.3,100,20,LINE02,True
SEC022,STN012,STN011,1.3,100,20,LINE02,True
SEC023,STN012,STN013,1.2,95,20,LINE02,True
SEC024,STN013,STN012,1.2,95,20,LINE02,True
SEC025,STN013,STN014,2.8,180,20,LINE02,True
SEC026,STN014,STN013,2.8,180,20,LINE02,True
SEC027,STN015,STN016,2.0,110,20,LINE03,True
SEC028,STN016,STN015,2.0,110,20,LINE03,True
SEC029,STN016,STN017,3.5,190,20,LINE03,True
SEC030,STN017,STN016,3.5,190,20,LINE03,True
SEC031,STN017,STN019,2.6,140,20,LINE03,True
SEC032,STN019,STN017,2.6,140,20,LINE03,True
SEC033,STN019,STN018,8.5,400,20,LINE03,True
SEC034,STN018,STN019,8.5,400,20,LINE03,True
SEC035,STN019,STN020,2.2,150,20,LINE04,True
SEC036,STN020,STN019,2.2,150,20,LINE04,True
SEC037,STN015,STN017,6.5,240,0,LINE05,True
SEC038,STN017,STN015,6.5,240,0,LINE05,True
SEC039,STN017,STN018,10.0,360,0,LINE05,True
SEC040,STN018,STN017,10.0,360,0,LINE05,True"""

        stations = HKMetroDataLoader.load_stations(stations_text)
        lines = HKMetroDataLoader.load_lines(lines_text)
        segments = HKMetroDataLoader.load_sections(sections_text)

        for segment in segments:
            line_id = segment['line_id']
            from_station = segment['from_station_id']
            to_station = segment['to_station_id']
            if line_id in lines:
                if from_station not in lines[line_id]['stations']:
                    lines[line_id]['stations'].append(from_station)
                if to_station not in lines[line_id]['stations']:
                    lines[line_id]['stations'].append(to_station)
            if from_station in stations and line_id not in stations[from_station]['lines']:
                stations[from_station]['lines'].append(line_id)
            if to_station in stations and line_id not in stations[to_station]['lines']:
                stations[to_station]['lines'].append(line_id)
            if line_id in lines:
                lines[line_id]['segments'].append(segment)

        for station_id, station in stations.items():
            station['passenger_flow'] = {}
            for hour in range(6, 24):
                base = 200
                if "中环" in station['name_cn'] or "Central" in station['name_en']:
                    base = 800
                elif "旺角" in station['name_cn'] or "Mong Kok" in station['name_en']:
                    base = 700
                elif "尖沙咀" in station['name_cn'] or "Tsim Sha Tsui" in station['name_en']:
                    base = 600
                elif "荃湾" in station['name_cn'] or "Tsuen Wan" in station['name_en']:
                    base = 400
                if 7 <= hour <= 9:
                    flow = int(base * 1.8)
                elif 17 <= hour <= 19:
                    flow = int(base * 1.5)
                else:
                    flow = int(base * 0.7)
                station['passenger_flow'][str(hour)] = flow

        for line_id, line in lines.items():
            total_distance = 0
            unique = set()
            for seg in line['segments']:
                key = tuple(sorted([seg['from_station_id'], seg['to_station_id']]))
                if key not in unique:
                    total_distance += seg['distance_km']
                    unique.add(key)
            line['total_distance_km'] = round(total_distance, 1)
            line['station_count'] = len(line['stations'])
            if line_id == 'LINE05':
                line['min_headway'] = 300
                line['max_headway'] = 1200
                line['train_capacity'] = 800
            elif line_id == 'LINE04':
                line['min_headway'] = 300
                line['max_headway'] = 1800
                line['train_capacity'] = 1000
            else:
                line['min_headway'] = 120
                line['max_headway'] = 600
                line['train_capacity'] = 1500

        print(f"✅ 网络构建完成: {len(stations)}个车站, {len(lines)}条线路")
        return {'stations': stations, 'lines': lines, 'segments': segments}

    @staticmethod
    def build_context(network: Dict) -> str:
        ctx = []
        lines = network['lines']
        ctx.append("香港地铁线路信息：")
        for lid, ln in lines.items():
            ctx.append(f"  {ln['name_cn']} ({ln['name_en']})，列车容量 {ln['train_capacity']} 人，"
                       f"最小间隔 {ln['min_headway']} 秒，最大间隔 {ln['max_headway']} 秒，"
                       f"车站数 {ln['station_count']}，全长 {ln['total_distance_km']} km。")
        ctx.append("常见运行参数：一般站间运行时间约 100~200 秒，停站时间约 20~30 秒。")
        ctx.append("高峰客流放大系数 1.8（早高峰），1.5（晚高峰）。")
        return "\n".join(ctx)

# ==================== 模型库 ModelLibrary ====================
class ModelLibrary:
    def __init__(self):
        self.templates = {
            ProblemType.TIMETABLING.value: self._build_timetabling_model(),
            ProblemType.HEADWAY.value: self._build_headway_model(),
            ProblemType.CAPACITY.value: self._build_capacity_model(),
            ProblemType.DELAY.value: self._build_delay_model(),
            ProblemType.RESCHEDULING.value: self._build_rescheduling_model(),
            ProblemType.ROLLING_STOCK.value: self._build_rolling_stock_model(),
            ProblemType.CIRCULATION.value: self._build_circulation_model(),
            ProblemType.CREW.value: self._build_crew_model(),
            ProblemType.PASSENGER_ASSIGNMENT.value: self._build_assignment_model(),
            ProblemType.CROWDING.value: self._build_crowding_model(),
            ProblemType.ENERGY.value: self._build_energy_model(),
            ProblemType.REGENERATIVE.value: self._build_energy_model(),
            ProblemType.NETWORK_FLOW.value: self._build_network_flow_model(),
            ProblemType.INTEGRATED_MTR.value: self._build_integrated_mtr_model(),
        }
    
    def get_model(self, problem_type: str) -> Optional[Dict]:
        return self.templates.get(problem_type)
    
    def _build_timetabling_model(self) -> Dict:
        return {"name": "Periodic Timetabling", "variables": ["t_arr[i]", "t_dep[i]"], "objective": "min_total_delay", "constraints": ["running_time", "dwell_time", "headway"]}
    def _build_headway_model(self) -> Dict:
        return {"name": "Headway Optimization", "variables": ["h"], "objective": "min_waiting_time", "formula": "Q*h/2", "constraints": ["capacity", "fleet_limit", "headway_bound", "cycle_constraint"]}
    def _build_capacity_model(self) -> Dict:
        return {"name": "Capacity Allocation", "variables": ["N", "h"], "objective": "min_operating_cost_plus_wait_cost", "constraints": ["load_factor"]}
    def _build_delay_model(self) -> Dict:
        return {"name": "Delay Propagation", "variables": ["D[i]"], "objective": "min_total_delay", "constraints": ["delay_recursion"]}
    def _build_rescheduling_model(self) -> Dict:
        return {"name": "Rescheduling", "variables": ["x[i,j]"], "objective": "min_total_delay", "constraints": ["headway", "conflict_resolution"]}
    def _build_rolling_stock_model(self) -> Dict:
        return {"name": "Rolling Stock Scheduling", "variables": ["x[i,j]"], "objective": "min_fleet_size", "constraints": ["vehicle_connectivity"]}
    def _build_circulation_model(self) -> Dict:
        return {"name": "Train Circulation", "variables": ["x[i,j]"], "objective": "min_fleet_size", "constraints": ["turnback_time"]}
    def _build_crew_model(self) -> Dict:
        return {"name": "Crew Scheduling", "variables": ["x[i]"], "objective": "min_crew_cost", "constraints": ["cover_all_trips"]}
    def _build_assignment_model(self) -> Dict:
        return {"name": "Passenger Assignment", "variables": ["flow[r]"], "objective": "user_equilibrium", "constraints": ["flow_conservation"]}
    def _build_crowding_model(self) -> Dict:
        return {"name": "Station Crowding", "variables": ["lambda", "mu"], "objective": "min_waiting_time", "constraints": ["queue_balance"]}
    def _build_energy_model(self) -> Dict:
        return {"name": "Energy Efficient Control", "variables": ["traction", "coast", "brake"], "objective": "min_energy", "constraints": []}
    def _build_network_flow_model(self) -> Dict:
        return {"name": "Min Cost Flow", "variables": ["x[i,j]"], "objective": "min_cost", "constraints": ["flow_balance"]}
    def _build_integrated_mtr_model(self) -> Dict:
        return {"name": "Integrated MTR", "objective": "PassengerDelay + OperatingCost", "submodels": ["timetabling", "headway", "capacity", "rolling_stock", "delay", "rescheduling"]}

# ==================== Math Agent（带消融实验开关） ====================
class MathAgent:
    def __init__(self, api, network_context: str, ablation_mode='full'):
        self.api = api
        self.context = network_context
        self.ablation_mode = ablation_mode
        self.system_prompt = "你是一位运筹优化专家，擅长将地铁调度问题转化为数学规划模型。请严格按照给定的模板输出。"

    def generate_model(self, query: str, params: Dict, template: Dict) -> str:
        try:
            from knowledge_base import PolicyKB
            recovery_factor = PolicyKB.get("recovery_factor", 0.6)
            delay_weight = PolicyKB.get("delay_weight", 0.8)
            energy_weight = PolicyKB.get("energy_weight", 0.2)
        except Exception:
            recovery_factor = 0.6
            delay_weight = 0.8
            energy_weight = 0.2
        
        # 💥 消融开关：如果关闭 rules，删掉 6-14 的所有核心规则
        if self.ablation_mode == 'no_rules':
            rules_text = """
规则：
1. 禁止发明新参数。
2. 禁止假设值。
3. 如果参数缺失，将其放入“缺失参数”部分。
4. 仅使用明确给出的参数与模型库参数。
"""
            print("   ⚠️ [Ablation] Domain Rules 已禁用 (退化为普通大模型)")
        else:
            rules_text = """
规则：
1. 禁止发明新参数。
2. 禁止假设值。
3. 如果参数缺失，将其放入“缺失参数”部分。
4. 仅使用明确给出的参数与模型库参数。
5. 不要创建线路长度、列车速度、循环时间等未知参数。
6. 目标函数设定规则（极度重要）：
- 如果当前问题类型是 headway（发车间隔优化），目标函数【必须】是 最小化等待时间 (Minimize = Q * h / 2)。
- 如果当前问题类型是 capacity（运力/班次配置），目标函数【必须】是 最小化发车班次 (Minimize = frequency)。
- 【降级规则】：如果问题属于纯计算型，【绝对禁止】画蛇添足地设置复杂的最大化/最小化目标函数和多余的决策变量！请直接使用单一等式赋值计算。
7. 命名规范：发车间隔变量严格命名为 `headway`；发车班次严格命名为 `frequency`。【极度重要】：求发车间隔时，【绝对禁止】把 frequency 定义为独立的决策变量并使用 frequency * headway == 3600！必须直接使用线性化形式（如 3600 * C >= Q * headway）。
8. 延误消除阈值：遇到“延误完全消除”时，约束设为 `D_n <= 1`，绝对禁止使用 `D_n == 0`。
9. 非负性处理：延误时间不可为负。若存在主动压缩策略，必须用 `>=` 不等式代替等式。
10. 阶段索引规范（极其重要）：初始状态（如“初始延误”）必须严格定义为第 0 阶段（即 D_0）。
11. 领域隔离（Domain Isolation）：纯粹的延误推演问题（求残留延误、评估挽回策略等），【绝对禁止】生成与客流需求（Q）、列车容量（C）相关的运力约束，严禁画蛇添足。
12. 物理与逻辑常识底线：任何代表时间、延误、客流的变量，必须强制加上 >= 0 的边界约束；严禁凭空捏造换算公式！
13. 拥挤阈值逻辑（极其重要）：【只有】明确要求“求最大滞留人数不超过X的临界发车间隔”时，才 最大化 headway。常规的发车间隔优化题绝对禁止最大化发车间隔。
14. 排队系统稳定性诊断：如果题目仅给出“到达率”和“能力/疏散率”并询问“状态”，这属于理论诊断题。必须建立连续变量 `t >= 0` 和 `queue_length >= 0`，并 `Maximize = queue_length`，让求解器自行返回 Unbounded。
"""

        prompt = f"""
你是运筹优化专家。
将以下优化问题转化为严格的数学模型。

{rules_text}

运营策略参数：
- 延误恢复系数 recovery_factor = {recovery_factor}
- 多目标权重：延误权重 delay_weight = {delay_weight}，能源权重 energy_weight = {energy_weight}

延误传播模型（基准公式）：
D_{{k+1}} = recovery_factor * D_k   （其中 recovery_factor = {recovery_factor}）
⚠️ 策略拓展与非负陷阱：
如果题目中提供了主动控制策略，绝对禁止使用等式 (`=`)！你必须用**大于等于的不等式**来代替 `max(0, ...)` 逻辑：
正确写法示例：`D_{{k+1}} >= recovery_factor * D_k - 压缩量`

输出格式（必须严格按此顺序）：
## Decision Variables
## Objective Function
## Constraints
(必须在开头标注 [HARD] 或 [SOFT])
## Missing Parameters

问题：
{query}
已知参数：
{json.dumps(params, indent=2, ensure_ascii=False)}
模型模板：
{json.dumps(template, indent=2)}
注意：只输出上述四个部分。
"""
        return self.api.query(prompt, self.system_prompt, temperature=0.1)
    
# ==================== 线性化辅助 ====================
class ModelLinearizer:
    def linearize(self, model_text: str) -> str:
        pattern = r'3600\s*/\s*h\s*\*\s*C\s*>=\s*Q'
        replacement = '3600*C >= Q*h'
        model_text = re.sub(pattern, replacement, model_text, flags=re.IGNORECASE)
        pattern2 = r'3600/h\*C>=Q'
        model_text = re.sub(pattern2, '3600*C >= Q*h', model_text, flags=re.IGNORECASE)
        return model_text

# ==================== 优化规格构建器 (修复物理常识) ====================
class OptimizationSpecBuilder:
    def build(self, problem_type: str, params: Dict) -> Dict:
        spec = {
            "problem_type": problem_type,
            "parameters": params,
            "missing_params": []
        }
        if problem_type == "headway":
            required = ["Q", "C", "N", "h_min", "h_max"]
            for p in required:
                if p not in params or params[p] is None:
                    spec["missing_params"].append(p)
        if params.get("N") is None and "N" not in spec["missing_params"]:
            spec["missing_params"].append("N")
        if params.get("Q") is None and "Q" not in spec["missing_params"]:
            spec["missing_params"].append("Q")
        # ========== 修改点：物理常识修复 ==========
        if problem_type == "headway":
            spec["constraint_expressions"] = {
                "capacity": "3600 * C >= Q * h",
                "headway_bound": "h_min <= h <= h_max"
            }
            # 物理常识修复：只有在没有给定周转时间时，才使用简化的 1小时约束
            if params.get("cycle_time") is not None:
                T = params["cycle_time"] * 60
                spec["constraint_expressions"]["cycle_constraint"] = f"N * h >= {T}"
            else:
                spec["constraint_expressions"]["fleet_limit"] = "3600 / h <= N"
                
            spec["objective"] = {"type": "waiting_time", "expression": "Q * h / 2"}
        # =========================================
        return spec

# ==================== Code Agent（强化变量提取，安全处理无解，支持非凸） ====================
class CodeAgent:
    def __init__(self, api: DeepSeekAPI):
        self.api = api
        self.system_prompt = "你是一位 Gurobi 优化专家。请严格按照要求生成代码。"

    def _strip_code_fences(self, code: str) -> str:
        # 使用更强健的正则，彻底剥离可能存在的中文废话
        match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return code.strip()

    def generate_code(self, spec: Dict, math_model: str = "") -> str:
        extracted_params = spec.get("parameters", {})
        
        prompt = f"""
You are a Senior Operations Research Engineer and Gurobi Expert. 
Your task is to write a standalone, executable Python script using Gurobi to solve the specified optimization problem.

[CRITICAL RULES - STRICT COMPLIANCE REQUIRED]
1. OUTPUT FORMAT: Output ONLY valid, runnable Python code. Do NOT include any conversational text.
2. EXCEPTION HANDLING (MANDATORY): You MUST wrap the entire modeling and optimization logic inside a `try...except Exception as e:` block.
3. STRICT ALIGNMENT: You MUST strictly implement the objective function EXACTLY as defined in the [MATH MODEL]. DO NOT use default multi-objective formulas unless requested.
4. HARD VS SOFT CONSTRAINTS: Respect the constraint labels in the [MATH MODEL]. [HARD] constraints MUST NOT have slack variables.
5. FINAL OUTPUT: The script MUST end by printing a single JSON string containing: 'status', 'answer', 'variables', and 'is_valid'.
   【极度重要 - variables 强制提取】：必须使用以下安全代码提取变量（绝对禁止在模型无解时强行提取 .X 导致崩溃）：
   vars_dict = {{}}
   if model.status == GRB.OPTIMAL:
       vars_dict = {{v.VarName: v.X for v in model.getVars()}}
   然后将 `{{"variables": vars_dict}}` 放入 JSON。
   【极度重要 - answer】：'answer' 字段必须填入核心变量的值（如 vars_dict.get('headway') 或 vars_dict.get('h')）。【绝对禁止】将 model.ObjVal 填入 answer！
6. 状态解析与参数设置 (极其重要): 
- 在调用 `model.optimize()` 之前，必须添加一行 `model.setParam('DualReductions', 0)`。
- 【极度重要】：必须添加一行 `model.setParam('NonConvex', 2)` 强制允许 Gurobi 求解非凸二次约束！
- 提取结果时，绝对禁止把所有非最优状态写死为 'Infeasible'！必须严谨判断：
  if model.status == GRB.OPTIMAL: status = 'Optimal'
  elif model.status == GRB.UNBOUNDED: status = 'Unbounded'
  elif model.status == GRB.INFEASIBLE: status = 'Infeasible'
  else: status = 'Infeasible'
  
[EXTRACTED PARAMETERS]
{json.dumps(extracted_params, ensure_ascii=False, indent=2)}

[PROBLEM SPECIFICATION]
{json.dumps(spec, ensure_ascii=False, indent=2)}

[MATH MODEL]
{math_model}

Write the complete Python script now:
"""
        raw_code = self.api.query(prompt, self.system_prompt, temperature=0.0)
        return self._strip_code_fences(raw_code)

# ==================== DebuggingAgent ====================
class DebuggingAgent:
    def __init__(self, api: DeepSeekAPI, math_agent: MathAgent, code_agent: CodeAgent, ablation_mode='full'):
        self.api = api
        self.math_agent = math_agent
        self.code_agent = code_agent
        self.ablation_mode = ablation_mode
        self.max_attempts = 5
        
        # 💥 消融开关：如果关闭 debug，只给 1 次机会，直接生算，不自愈
        if self.ablation_mode == 'no_debug':
            self.max_attempts = 1
            
        self.timeout = 30

    def _run_code_safely(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        code = re.sub(r'```python\s*', '', code, flags=re.I)
        code = re.sub(r'```\s*', '', code)
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(code)
            temp_file_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path],
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print("\n========= GENERATED CODE =========")
            print(code)
            print("==================================\n")
            print("\n========== RAW STDOUT ==========")
            print(result.stdout)
            print("================================\n")
            if result.returncode == 0:
                output_str = result.stdout
                json_lines = [line.strip() for line in output_str.splitlines() 
                              if line.strip().startswith("{") and line.strip().endswith("}")]
                if not json_lines:
                    return {'status': 'Error', 'error': f"No JSON found in stdout.\nSTDOUT=\n{output_str}"}
                try:
                    parsed = json.loads(json_lines[-1])
                    status = parsed.get("status")
                    if status == GRB.OPTIMAL:
                        parsed["status"] = "Optimal"
                    elif status == GRB.INFEASIBLE:
                        parsed["status"] = "Infeasible"
                    elif status == GRB.UNBOUNDED:
                        parsed["status"] = "Unbounded"
                    return parsed
                except json.JSONDecodeError as e:
                    return {'status': 'Error', 'error': f"JSON parse error: {e}\nLine: {json_lines[-1]}"}
            else:
                return {'status': 'Error', 'error': f"代码运行崩溃，错误信息:\n{result.stderr}"}
        except subprocess.TimeoutExpired:
            return {'status': 'Error', 'error': f'执行超时：Gurobi 求解或代码执行时间超过了 {timeout} 秒限制。'}
        except Exception as e:
            return {'status': 'Error', 'error': f'沙箱执行器遭遇未知异常: {str(e)}', 'traceback': traceback.format_exc()}
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    def _repair_code(self, old_code: str, error_msg: str, original_query: str) -> str:
        fix_prompt = f"""
The previous Gurobi optimization code failed. 

[ORIGINAL PROBLEM]
{original_query}

[PREVIOUS CODE]
{old_code}

[ERROR/OUTPUT RECEIVED]
{error_msg}

[CRITICAL RULES - STRICT COMPLIANCE REQUIRED]
1. OUTPUT FORMAT: Output ONLY valid, runnable Python code. Do NOT include any conversational text.
2. EXCEPTION HANDLING: Ensure the try-except block printing JSON is intact.
3. OUTPUT STATUS KEY: The JSON status MUST be strictly capitalized as "Optimal" or "Infeasible".

Please fix the code based on the error. For Nonlinear constraints (like dividing by a variable), linearize it mathematically (e.g., change `A >= B / h` to `A * h >= B`).
Return the completely fixed Python code:
"""
        new_code = self.code_agent.api.query(fix_prompt, self.code_agent.system_prompt, temperature=0.0)
        return self.code_agent._strip_code_fences(new_code)

    def execute_and_debug(self, initial_code: str, original_query: str) -> Dict[str, Any]:
        code = initial_code
        for attempt in range(1, self.max_attempts + 1):
            print(f"   🔄 尝试执行第 {attempt} 次...")
            result = self._run_code_safely(code, timeout=self.timeout)
            # ==================== 替换 DebuggingAgent 中的状态判断 ====================
            opt_status = str(result.get('status', '')).lower()
            
            is_optimal = opt_status in ['2', 'optimal']
            is_unbounded = opt_status in ['5', 'unbounded']
            is_infeasible = opt_status in ['3', '4', 'infeasible']
            
            if is_optimal or is_infeasible or is_unbounded:
                result['attempts'] = attempt
                if is_unbounded:
                    result['status'] = 'Unbounded'
                elif is_infeasible:
                    result['status'] = 'Infeasible'
                else:
                    result['status'] = 'Optimal'
                return result
            # =========================================================================
            
            else:
                error_msg = result.get('error', result.get('raw_output', '未知错误'))
                print(f"   ❌ 执行失败（第 {attempt} 次）: {error_msg}")
                if result.get("traceback"):
                    print("\n========== TRACEBACK ==========")
                    print(result["traceback"])
                    print("================================\n")
                if attempt < self.max_attempts:
                    print("   🔧 代码自修复...")
                    code = self._repair_code(code, error_msg, original_query)
                else:
                    print("   ⚠️ 已达最大尝试次数，求解失败。")
                    result['attempts'] = attempt
                    result['status'] = 'Failed'
                    return result
        return {'status': 'Failed', 'attempts': self.max_attempts}

# ==================== 硬性约束校验器 (带推演豁免版) ====================
class HeadwayConstraintVerifier:
    @staticmethod
    def verify_headway(result: Dict[str, Any], params: Dict[str, Any]) -> Tuple[bool, str]:
        opt_status = result.get('status')
        is_optimal = (opt_status in [2, 'Optimal', 'OPTIMAL']) or (opt_status == GRB.OPTIMAL)
        if not is_optimal:
            return False, f"求解状态非 Optimal: {result.get('status')}"
            
        # ==================== 校验豁免补丁 ====================
        # 判断是否为纯延误状态推演 (变量里包含 D1, D2... 且不涉及复杂的列车排班变量)
        variables = result.get("variables", {})
        is_pure_state_math = any(key.startswith('D') for key in variables.keys())
        
        h = result.get('headway') or result.get('optimal_h')
        
        if h is None:
            if is_pure_state_math:
                return True, "状态推演约束校验通过 (无需 headway)" # 触发豁免，返回 True
            return False, "结果中未找到 headway 或 optimal_h"
        # ======================================================

        Q = params.get('Q')
        C = params.get('C')
        h_min = params.get('h_min', 120)
        h_max = params.get('h_max', 600)
        
        if h < h_min - 1e-6:
            return False, f"发车间隔 {h:.1f} 秒 小于最小允许间隔 {h_min} 秒"
        if h > h_max + 1e-6:
            return False, f"发车间隔 {h:.1f} 秒 大于最大允许间隔 {h_max} 秒"
        if Q is not None and C is not None:
            required_h = (3600 * C) / Q
            if h > required_h + 1e-6:
                return False, f"发车间隔 {h:.1f} 秒 大于满足运力所需的最大间隔 {required_h:.1f} 秒"
                
        return True, "通过"

# ==================== Solver Statistics Collector ====================
class SolverStatisticsCollector:
    @staticmethod
    def collect(result: Dict) -> Dict:
        return {
            "status": result.get("status"),
            "objective": result.get("objective"),
            "runtime": result.get("runtime"),
            "num_vars": result.get("num_vars"),
            "num_constraints": result.get("num_constraints"),
            "mip_gap": result.get("mip_gap")
        }

# ==================== Model Explainability Agent ====================
class ModelExplainabilityAgent:
    def generate_model_selection(self, problem_type, template):
        return {"problem_type": problem_type, "selected_model": template.get("name"), "reason": f"系统识别问题类型为 {problem_type}，因此选用 {template.get('name')} 模型。"}
    def generate_constraint_explanation(self, spec):
        explanations = []
        cons = spec.get("constraint_expressions", {})
        if "capacity" in cons:
            explanations.append({"constraint": cons["capacity"], "meaning": "线路运力必须满足客流需求"})
        if "headway_bound" in cons:
            explanations.append({"constraint": cons["headway_bound"], "meaning": "发车间隔必须位于允许范围内"})
        if "fleet_limit" in cons:
            explanations.append({"constraint": cons["fleet_limit"], "meaning": "列车数量不能超过可用车底"})
        if "cycle_constraint" in cons:
            explanations.append({"constraint": cons["cycle_constraint"], "meaning": "列车周转约束：可用列车数乘以发车间隔必须大于等于周期时间"})
        return explanations
    def generate_solver_explanation(self, result):
        return {"solver": "Gurobi", "status": result.get("status"), "runtime": result.get("runtime"), "interpretation": "求解器成功找到全局最优解" if result.get("status") == "Optimal" else "求解失败或无可行解"}
    def generate_assumption_explanation(self, params):
        assumptions = []
        if "h_min" in params:
            assumptions.append({"parameter": "h_min", "value": params["h_min"], "source": "MTR Default", "reason": "线路默认最小发车间隔"})
        if "h_max" in params:
            assumptions.append({"parameter": "h_max", "value": params["h_max"], "source": "MTR Default", "reason": "线路默认最大发车间隔"})
        if "N" in params:
            assumptions.append({"parameter": "N", "value": params["N"], "source": "System Default", "reason": "默认可用车底数量"})
        if "cycle_time" in params:
            assumptions.append({"parameter": "cycle_time", "value": params["cycle_time"], "source": "Infrastructure KB", "reason": "线路周期时间"})
        return assumptions
    def generate_operational_recommendation(self, problem_type, result, params):
        recs = []
        if problem_type in ["headway", "capacity"]:
            h = result.get("headway")
            if h and params.get("Q") and params.get("C"):
                tph = 3600 / h
                load_factor = (params["Q"] / tph) / params["C"]
                if load_factor < 0.4:
                    recs.append("当前运力明显富余，可适当增大发车间隔降低运营成本。")
                elif load_factor > 0.9:
                    recs.append("线路接近满载，建议增开列车。")
        elif problem_type in ["delay", "rescheduling"]:
            delays = result.get("delays", [])
            if delays and isinstance(delays, list) and len(delays) > 0:
                if delays[-1] <= 1.0:
                    recs.append(f"延误在传播 {len(delays)} 次后已基本消除，建议在此期间加强沿线车站客流疏导。")
                else:
                    recs.append(f"系统经过 {len(delays)} 次传播后延误未能完全消除 (剩余 {delays[-1]}秒)，建议立即启动应急备用车底或缩短后续列车发车间隔。")
        else:
            if result.get("status") in ["Optimal", 2]:
                obj_val = result.get("objective_value") or result.get("objective")
                if obj_val is not None:
                    recs.append(f"系统已针对 {problem_type} 找到理论最优方案（目标值 {obj_val}），建议调度员结合实际情况执行。")
                else:
                    recs.append(f"系统已针对 {problem_type} 找到理论最优方案，建议调度员结合实际情况执行。")
            elif result.get("status") in ["Infeasible", 3, 4, 5]:
                recs.append("当前设定的参数（如客流与最小发车间隔）存在数学冲突，导致无法排班/满足运力。建议放宽约束条件重试。")
        return recs
    def generate_kpi_analysis(self, problem_type, result, params):
        kpi = {}
        if problem_type in ["headway", "capacity"]:
            h = result.get("headway")
            if h and params.get("Q") and params.get("C"):
                tph = 3600 / h
                capacity_hour = tph * params["C"]
                load_factor = params["Q"] / capacity_hour
                kpi = {
                    "train_per_hour": round(tph, 2),
                    "capacity_per_hour": round(capacity_hour, 2),
                    "load_factor": round(load_factor, 4),
                    "average_waiting_time": round(h / 2, 2),
                    "spare_capacity": round(capacity_hour - params["Q"], 2)
                }
        elif problem_type == "delay":
            delays = result.get("delays", [])
            if delays and isinstance(delays, list):
                kpi = {
                    "initial_delay_sec": delays[0] if len(delays) > 0 else 0,
                    "final_delay_sec": delays[-1] if len(delays) > 0 else 0,
                    "delay_stages_count": len(delays),
                    "objective_value": result.get("objective_value", 0)
                }
        else:
            if result.get("status") in ["Optimal", 2]:
                obj_val = result.get("objective_value") or result.get("objective")
                if obj_val is not None:
                    kpi = {
                        "optimization_status": "Global Optimal",
                        "objective_value": round(float(obj_val), 2),
                        "model_efficiency": "High"
                    }
                else:
                    kpi = {"status": "Solved but no objective value returned"}
            elif result.get("status") in ["Infeasible", 3, 4, 5]:
                kpi = {
                    "optimization_status": "Mathematically Infeasible",
                    "conflict_reason": "Parameters contradict physical or operational limits."
                }
        return kpi
    def generate_shadow_price_explanation(self, result):
        return {
            "capacity_dual": result.get("capacity_dual"),
            "fleet_dual": result.get("fleet_dual"),
            "headway_dual": result.get("headway_dual")
        }
    def generate_sensitivity_analysis(self, params, result):
        Q = params.get("Q")
        if not Q:
            return {}
        return {"-20%": round(Q * 0.8), "-10%": round(Q * 0.9), "base": Q, "+10%": round(Q * 1.1), "+20%": round(Q * 1.2)}
    def explain_adaptation(self, adaptation_trace):
        if not adaptation_trace:
            return {"adaptation": "No Adaptation", "reason": "基础模型已满足需求"}
        return adaptation_trace

# ==================== Post-Hoc Explainability Agent ====================
class ExplainabilityAgent:
    def explain(self, query: str, params: Dict, result: Dict) -> Dict:
        h = result.get("headway") or result.get("optimal_h")
        explanation = []
        if h is not None:
            explanation.append(f"求解器得到最优发车间隔 {h:.1f} 秒。")
        Q = params.get("Q")
        C = params.get("C")
        if Q and C and h:
            capacity_limit = 3600 * C / Q
            explanation.append(f"容量约束允许的最大间隔为 {capacity_limit:.1f} 秒。")
            h_min = params.get("h_min", 120)
            if abs(h - h_min) < 1e-6:
                explanation.append("最优解等于最小发车间隔，因此最小间隔约束成为Binding Constraint。")
        return {"type": "Explainability", "content": explanation}

# ==================== Controlled Adaptation Agent ====================
class ControlledAdaptationAgent:
    def simulate(self, params: Dict) -> Dict:
        Q = params.get("Q")
        C = params.get("C")
        if Q is None or C is None:
            return {"status": "Missing Parameters"}
        new_Q = int(Q * 1.2)
        max_h = 3600 * C / new_Q
        return {"scenario": "客流增加20%", "old_Q": Q, "new_Q": new_Q, "expected_max_headway": round(max_h, 2)}

# ==================== Recommendation Agent ====================
class RecommendationAgent:
    def generate_recommendations(self, problem_type: str, params: Dict, result: Dict) -> List[Dict]:
        recommendations = []
        if problem_type == "headway":
            Q = params.get("Q")
            C = params.get("C")
            h = result.get("headway") or result.get("optimal_h")
            if not (Q and C and h):
                return recommendations
            tph = 3600 / h
            load_factor = (Q / tph) / C
            if load_factor < 0.4:
                recommendations.append({"type": "increase_headway", "current_h": h, "candidate_h": 180, "reason": "运力富余"})
                recommendations.append({"type": "increase_headway", "current_h": h, "candidate_h": 240, "reason": "进一步降低运营成本"})
            elif load_factor > 0.9:
                recommendations.append({"type": "decrease_headway", "current_h": h, "candidate_h": max(120, h-30), "reason": "缓解拥挤"})
        return recommendations

# ==================== Sensitivity Analyzer ====================
class SensitivityAnalyzer:
    def analyze_headway(self, params: Dict) -> List[Dict]:
        results = []
        Q = params["Q"]
        C = params["C"]
        h_min = params.get("h_min", 120)
        h_max = params.get("h_max", 600)
        step = 30
        h = h_min
        candidate_headways = []
        while h <= h_max:
            candidate_headways.append(h)
            h += step
        for h in candidate_headways:
            tph = 3600 / h
            capacity = tph * C
            waiting_time = Q * h / 2
            load_factor = Q / capacity
            fleet_usage = tph
            results.append({"headway": h, "waiting_time": round(waiting_time, 2), "load_factor": round(load_factor, 3), "fleet_usage": round(fleet_usage, 2)})
        return results

# ==================== Operation Plan Agent ====================
class OperationPlanAgent:
    def build_plan(self, sensitivity_results: List[Dict]) -> Dict:
        if not sensitivity_results:
            return {}
        service_plan = min(sensitivity_results, key=lambda x: x["waiting_time"])
        cost_plan = min(sensitivity_results, key=lambda x: x["fleet_usage"])
        balance_plan = min(sensitivity_results, key=lambda x: abs(x["load_factor"] - 0.7))
        return {"service_first": service_plan, "balanced": balance_plan, "cost_first": cost_plan}

# ==================== Scenario Evaluator ====================
class ScenarioEvaluator:
    def evaluate(self, problem_type: str, params: Dict, recommendations: List[Dict]) -> List[Dict]:
        Q = params.get("Q")
        C = params.get("C")
        if Q is None:
            return [{"status": "Skipped", "reason": "Passenger demand Q unavailable"}]
        if C is None:
            return [{"status": "Skipped", "reason": "Capacity C unavailable"}]
        scenarios = []
        if problem_type != "headway":
            return scenarios
        for rec in recommendations:
            h = rec["candidate_h"]
            tph = 3600 / h
            capacity = tph * C
            waiting_time = Q * h / 2
            load_factor = Q / capacity
            scenarios.append({"scenario": f"h={h}s", "headway": h, "waiting_time": round(waiting_time, 2), "capacity": round(capacity, 2), "load_factor": round(load_factor, 3), "recommendation": rec["reason"]})
        return scenarios

# ==================== Scenario Generator ====================
class ScenarioGenerator:
    def generate(self, recommendations: List[Dict]) -> List[Dict]:
        scenarios = []
        for rec in recommendations:
            scenarios.append({"scenario_name": f"Scenario_{rec['candidate_h']}", "type": rec["type"], "headway": rec["candidate_h"]})
        return scenarios

# ==================== ReOptimization Agent ====================
class ReOptimizationAgent:
    def resolve_headway(self, params: Dict, headway: float) -> Dict:
        Q = params["Q"]
        C = params["C"]
        tph = 3600 / headway
        capacity = tph * C
        waiting_time = Q * headway / 2
        load_factor = Q / capacity
        operating_cost = tph
        return {"headway": headway, "waiting_time": round(waiting_time, 2), "capacity": round(capacity, 2), "load_factor": round(load_factor, 3), "operating_cost": round(operating_cost, 2)}

# ==================== Impact Analyzer ====================
class ImpactAnalyzer:
    def compare(self, base_result: Dict, scenario_result: Dict) -> Dict:
        base_obj = base_result.get("objective", 0)
        base_val = base_obj if base_obj is not None else 0
        scenario_val = scenario_result.get("waiting_time")
        if scenario_val is None:
            delta_waiting = 0
        else:
            delta_waiting = scenario_val - base_val
        return {
            "delta_waiting_time": delta_waiting,
            "delta_load_factor": scenario_result.get("load_factor", 0),
            "delta_operating_cost": scenario_result.get("operating_cost", 0)
        }

# ==================== Pareto Analyzer ====================
class ParetoAnalyzer:
    def analyze(self, sensitivity_results: List[Dict]) -> List[Dict]:
        frontier = []
        for row in sensitivity_results:
            dominated = False
            for other in sensitivity_results:
                if (other["waiting_time"] <= row["waiting_time"] and other["fleet_usage"] <= row["fleet_usage"] and other != row):
                    dominated = True
                    break
            if not dominated:
                frontier.append(row)
        return frontier

# ==================== Model Feasibility Agent ====================
class ModelFeasibilityAgent:
    def __init__(self):
        self.library = ParameterRequirementLibrary()
    def evaluate(self, problem_type: str, params: Dict) -> Dict:
        required = self.library.get_required_params(problem_type)
        missing = [p for p in required if params.get(p) is None]
        completeness = 100 if not required else round(100 * (len(required) - len(missing)) / len(required), 1)
        return {"feasible": len(missing) == 0, "required_params": required, "missing_params": missing, "completeness_score": completeness}

# ==================== 软警告检查器 ====================
class SolutionVerifier:
    def verify(self, problem_type: str, params: Dict, result: Dict) -> List[str]:
        warnings = []
        if problem_type == "headway":
            Q = params.get("Q")
            C = params.get("C")
            h = result.get('headway') or result.get('optimal_h')
            if Q and C and h:
                tph = 3600 / h
                load_factor = (Q / tph) / C
                if load_factor < 0.4:
                    warnings.append("运力明显富余，存在过度发车（负载率低于40%）")
                elif load_factor > 0.9:
                    warnings.append("运力接近饱和，负载率超过90%，建议增加运力")
        return warnings

# ==================== 缺失参数报告生成函数 ====================
def generate_missing_parameter_report(query: str, problem_type: str, feasibility: Dict) -> str:
    report = f"""
{'='*80}
📊 Metro Optimization Analysis Report
{'='*80}

用户问题:
{query}

识别问题类型:
{problem_type}

模型状态:
❌ 无法求解

原因:
关键参数缺失

参数完整度:
{feasibility['completeness_score']}%

需要参数:

{chr(10).join(['- ' + x for x in feasibility['required_params']])}

缺失参数:

{chr(10).join(['- ' + x for x in feasibility['missing_params']])}

建议:

请补充缺失参数后重新求解。

系统不会自动假设参数，
以保证 Explainable Reasoning
与 Controlled Adaptation。

{'='*80}
"""
    return report

# ==================== 报告生成器 (高阶业务逻辑版 + 去幻觉) ====================
class MathReportGenerator:
    def __init__(self, api: DeepSeekAPI):
        self.api = api

    def generate_report(
        self,
        query: str,
        math_model: str,
        optimization_result: Dict,
        model_selection: Dict,
        constraint_explanation: List,
        solver_explanation: Dict,
        assumption_explanation: List,
        operational_recommendation: List,
        kpi_analysis: Dict,
        shadow_price: Dict,
        sensitivity: Dict,
        recommendations: Optional[List[Dict]] = None,
        scenario_results: Optional[List[Dict]] = None,
        sensitivity_results: Optional[List[Dict]] = None,
        operation_plans: Optional[Dict] = None,
        pareto_front: Optional[List[Dict]] = None,
        reoptimization_results: Optional[List[Dict]] = None,
        impact_results: Optional[List[Dict]] = None
    ) -> str:
        timestamp = datetime.now()
        report_id = f"MATH-{timestamp.strftime('%Y%m%d-%H%M%S')}"
        status = optimization_result.get('status', 'Unknown')
        
        if status == 'Failed':
            error_detail = optimization_result.get('error', 'Unknown Error')
            report = f"""
{'='*80}
📊 地铁调度数学优化分析报告
{'='*80}
📄 报告ID: {report_id}
⏰ 生成时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🔍 用户查询: {query}
⚠️ 求解状态: 失败

求解器未成功得到可行解，
以下分析仅基于数学模型，
不代表优化结果。

详细错误信息：
{error_detail}

{'='*80}
📄 报告完毕
生成系统: 香港地铁调度优化系统 v19.1 (Enhanced Debug + Fallback)
{'='*80}
"""
            return report
        
        # ==================== 动态组装核心指标，消灭“发车间隔执念” ====================
        headway = optimization_result.get('headway') or optimization_result.get('optimal_h')
        extracted_ans = optimization_result.get('extracted_answer')
        problem_type = optimization_result.get("problem_type", "unknown")
        
        primary_metric_str = "N/A"
        if extracted_ans is not None:
            if problem_type == "headway":
                primary_metric_str = f"最优发车间隔: {extracted_ans} 秒"
            elif problem_type == "capacity":
                primary_metric_str = f"建议运力/班次: {extracted_ans}"
            elif problem_type in ["delay", "rescheduling"]:
                primary_metric_str = f"评估延误时间: {extracted_ans} 分钟"
            elif problem_type in ["crowding", "regenerative"]:
                primary_metric_str = f"系统评估比率: {extracted_ans}"
            else:
                primary_metric_str = f"核心结算结果: {extracted_ans}"
        elif "delays" in optimization_result and isinstance(optimization_result["delays"], list) and len(optimization_result["delays"]) > 0:
            primary_metric_str = f"初始延误: {optimization_result['delays'][0]} 分钟 -> 最终延误: {optimization_result['delays'][-1]} 分钟"
        elif "objective_value" in optimization_result:
            primary_metric_str = f"最优目标函数值: {optimization_result['objective_value']}"

        assumptions = optimization_result.get('assumptions', [])
        assumptions_str = ", ".join(assumptions) if assumptions else "无"
        warnings = optimization_result.get('warnings', [])
        
        # ==================== 智能可解释性引擎 (高阶业务逻辑版) ====================
        adaptation_trace = optimization_result.get("adaptation_trace", [])
        
        analysis_prompt = f"""
你是香港地铁(MTR)资深调度总长。请基于以下客观数据，生成极具业务价值的评估报告。

【用户查询】：{query}
【优化结果】：{json.dumps(optimization_result, ensure_ascii=False)}
【模型自适应轨迹】：{json.dumps(adaptation_trace, ensure_ascii=False)}

【极度重要的业务逻辑底线（严禁产生幻觉）】：
1. 绝对禁止跨领域套用专业术语！如果是一道延误(delay)题，算出的数值就是分钟数，绝对不能在报告中将其称为“发车间隔”；如果是运力(capacity)题，数值就是列车班次。
2. 如果状态为 "Infeasible"，说明存在物理矛盾。你必须点出矛盾所在，并给出真实的应对策略（如限流、抽调备用车），禁止建议修改安全上下限来迎合错误。
3. 请将【模型自适应轨迹】无缝融入分析中，解释“系统为什么这么考虑”。

请严格输出纯 JSON 格式：
{{
    "expert_analysis": "（200-300字）结合自适应轨迹，直击核心矛盾的深度因果分析。",
    "operational_recommendation": ["指令1：具体的SOP操作...", "指令2：..."],
    "kpi_analysis": {{"核心指标1": "业务解读", "核心指标2": "业务解读"}}
}}
"""
        # 3. 调用大模型并解析 JSON
        analysis_json = self.api.query_json(analysis_prompt, system_prompt="你是顶级的地铁调度分析专家。必须严格输出纯 JSON。")
        
        # 4. 提取并定义所有需要的变量（极其重要，防止报错）
        deep_analysis = analysis_json.get("expert_analysis", "专家分析生成失败或未返回预期格式。")
        
        # 覆盖之前传入的硬编码推荐
        if "operational_recommendation" in analysis_json and analysis_json["operational_recommendation"]:
            operational_recommendation = analysis_json["operational_recommendation"]
            
        if "kpi_analysis" in analysis_json and analysis_json["kpi_analysis"]:
            kpi_analysis = analysis_json["kpi_analysis"]

        # ==================== 重构：高管级业务摘要 + 工程师级技术附录 ====================
        
        report = f"""
{'='*80}
🚆 MTR 智能运筹调度分析报告 (AI-OR Agent)
{'='*80}
📄 报告ID: {report_id}  |  ⏰ 生成时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🔍 场景描述: {query}
📌 假设说明: {assumptions_str}

【第一部分：核心诊断与专家结论】 (面向管理层)
{'-'*80}
⚠️ 优化状态: **{status}**
🎯 核心指标: {primary_metric_str}

🧠 专家深度分析: 
{deep_analysis}

【第二部分：一线调度执行指令 (SOP)】 (面向调度员)
{'-'*80}
"""
        if operational_recommendation:
            for idx, rec in enumerate(operational_recommendation, 1):
                report += f"[{idx}] {rec}\n"
        else:
            report += "无待执行指令。\n"

        report += f"""
【第三部分：核心业务指标评估 (KPI)】 (面向管理层)
{'-'*80}
"""
        if kpi_analysis:
            for k, v in kpi_analysis.items():
                report += f"- **{k}**: {v}\n"
        else:
            report += "无核心指标。\n"

        # -------------------- 机械化参数与技术附录 --------------------
        report += f"""
【第四部分：底层数学模型与技术参数附录】 (面向算法工程师)
{'-'*80}
[4.1 数学模型摘要]
{math_model}

[4.2 原始求解器详细结果]
{json.dumps(optimization_result, ensure_ascii=False, indent=2)}

[4.3 模型选择 (Model Selection)]
{json.dumps(model_selection, ensure_ascii=False, indent=2)}

[4.4 约束解释 (Constraint Explanation)]
{json.dumps(constraint_explanation, ensure_ascii=False, indent=2)}

[4.5 求解器解析 (Solver Explanation)]
{json.dumps(solver_explanation, ensure_ascii=False, indent=2)}

[4.6 假设参数 (Assumption Explanation)]
{json.dumps(assumption_explanation, ensure_ascii=False, indent=2)}

[4.7 影子价格/对偶变量 (Shadow Price)]
{json.dumps(shadow_price, ensure_ascii=False, indent=2)}

[4.8 敏感性分析 (Demand Variation)]
{json.dumps(sensitivity, ensure_ascii=False, indent=2)}
"""
        if recommendations:
            report += f"\n[4.9 启发式建议评估 (Recommendation Evaluation)]\n"
            for r in recommendations:
                report += f"- 类型: {r['type']} | 建议间隔: {r['candidate_h']}s | 原因: {r['reason']}\n"
                
        if scenario_results and not (len(scenario_results) == 1 and "status" in scenario_results[0]):
            report += f"\n[4.10 场景评估 (Scenario Evaluation)]\n"
            for s in scenario_results:
                report += f"- 方案: {s['scenario']} | 等待时间: {s['waiting_time']:.0f} | 负载率: {s['load_factor']:.2f}\n"

        if sensitivity_results:
            report += f"\n[4.11 发车间隔敏感性 (Headway vs Performance)]\n"
            for row in sensitivity_results:
                report += f"- h={row['headway']}s | 等待: {row['waiting_time']:.0f} | 负载: {row['load_factor']:.2f} | 频率: {row['fleet_usage']:.1f}\n"

        if pareto_front:
            report += f"\n[4.12 帕累托前沿 (Pareto Frontier)]\n"
            for p in pareto_front:
                report += f"- h={p['headway']}s | 等待: {p['waiting_time']:.0f} | 负载: {p['load_factor']:.2f}\n"

        # 保留原汁原味的 Adaptation Trace
        report += f"\n[4.13 模型自适应原始轨迹 (Raw Adaptation Trace)]\n"
        if adaptation_trace:
            report += json.dumps(adaptation_trace, ensure_ascii=False, indent=2) + "\n"
        else:
            report += "未触发模型适应，使用基础模型。\n"

        if warnings:
            report += f"\n⚠️ [模型边缘风险提示]:\n"
            for w in warnings:
                report += f" - {w}\n"

        report += f"""
{'='*80}
✅ 报告完毕 | 生成系统: 香港地铁调度优化系统 v20
{'='*80}
"""
        return report

# ==================== Orchestrator ====================
class Orchestrator:
    # 增加一个默认值，防止以后传错参数
    def __init__(self, api_client: DeepSeekAPI = None, deepseek_api_key: str = None, ablation_mode: str = 'full'):
        self.ablation_mode = ablation_mode
        print(f"🧠 初始化 OR‑LLM‑Agent (模式: {self.ablation_mode.upper()})...")
        
        # 如果外面传了 api_client 对象，直接用；否则才根据 key 去创建
        if isinstance(api_client, DeepSeekAPI):
            self.api = api_client
        else:
            self.api = DeepSeekAPI(api_key=deepseek_api_key)
        self.api = api_client if api_client else DeepSeekAPI(deepseek_api_key)
        
        self.network = HKMetroDataLoader.create_network()
        self.network_context = HKMetroDataLoader.build_context(self.network)
        
        self.query_processor = QueryProcessor()
        self.intent_classifier = IntentClassifier(self.api)
        self.parameter_extractor = ParameterExtractor(self.api)
        self.feasibility_agent = ModelFeasibilityAgent()
        self.model_library = ModelLibrary()
        self.model_linearizer = ModelLinearizer()
        self.spec_builder = OptimizationSpecBuilder()
        self.headway_verifier = HeadwayConstraintVerifier()
        self.solution_verifier = SolutionVerifier()
        self.problem_classifier = ProblemClassifier(self.api)
        
        # 透传 ablation_mode 给 MathAgent 和 DebuggingAgent
        self.math_agent = MathAgent(self.api, self.network_context, self.ablation_mode)
        self.code_agent = CodeAgent(self.api)
        self.debug_agent = DebuggingAgent(self.api, self.math_agent, self.code_agent, self.ablation_mode)
        
        self.model_explainability_agent = ModelExplainabilityAgent()
        self.posthoc_explainability_agent = ExplainabilityAgent()
        self.adaptation_agent = ControlledAdaptationAgent()
        self.recommendation_agent = RecommendationAgent()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.operation_plan_agent = OperationPlanAgent()
        self.scenario_evaluator = ScenarioEvaluator()
        self.pareto_analyzer = ParetoAnalyzer()
        self.scenario_generator = ScenarioGenerator()
        self.reoptimization_agent = ReOptimizationAgent()
        self.impact_analyzer = ImpactAnalyzer()
        self.model_adaptation_agent = ModelAdaptationAgent()
        
        self.report_gen = MathReportGenerator(self.api)
        self.history = []

    def wrap_output(self, answer, explanation, reasoning, trace, variables, factors, constraint_check, report_string, is_valid=True, constraint_mapping=None):
        if constraint_mapping is None:
            constraint_mapping = {}
        return {
            "answer": answer,
            "explanation": explanation,
            "reasoning": reasoning,
            "trace": trace,
            "variables": variables,
            "factors": factors,
            "constraint_check": constraint_check,
            "constraint_mapping": constraint_mapping,
            "report": report_string,
            "is_valid": is_valid
        }

    def process(self, user_query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            start = datetime.now()
            print(f"\n📝 开始处理: {user_query}")

            query = self.query_processor.process(user_query)
            intent = self.intent_classifier.classify(query)
            problem_type_str = self.problem_classifier.classify(query)

            print(f"📌 Intent: {intent}")
            print(f"📌 Problem Type: {problem_type_str}")

            if intent == QueryIntent.EXPLAIN.value:
                if not self.history:
                    return {"success": False, "report": "无历史求解结果可解释"}
                last_result = self.history[-1]["result"]
                explain_result = self.posthoc_explainability_agent.explain(query, {}, last_result)
                return {"success": True, "report": json.dumps(explain_result, ensure_ascii=False, indent=2)}

            if intent == QueryIntent.ADAPT.value:
                if not self.history:
                    return {"success": False, "report": "无历史求解参数，无法进行适应性分析"}
                last_params = self.history[-1].get("params", {})
                adapt_result = self.adaptation_agent.simulate(last_params)
                return {"success": True, "report": json.dumps(adapt_result, ensure_ascii=False, indent=2)}

            required_params = ParameterRequirementLibrary.get_required_params(problem_type_str)
            extracted_params = self.parameter_extractor.extract(query, problem_type_str, required_params)

            if params:
                extracted_params.update(params)

            feasibility = self.feasibility_agent.evaluate(problem_type_str, extracted_params)
            print(f"📊 Parameter Completeness: {feasibility['completeness_score']}%")

            if not feasibility["feasible"]:
                print("❌ 参数不足，无法构建优化模型")
                missing_report = generate_missing_parameter_report(user_query, problem_type_str, feasibility)
                return self.wrap_output(
                    answer="MissingParameters",
                    explanation="参数缺失，无法求解。",
                    reasoning=["参数完整性检查未通过"],
                    trace=[],
                    variables=extracted_params,
                    factors=[],
                    constraint_check={},
                    report_string=missing_report,
                    is_valid=False
                )

            line_name = extracted_params.get("line")
            if line_name:
                line_info = InfrastructureKB.get_line(line_name)
                if line_info:
                    extracted_params.setdefault("run_time", line_info["run_time"])
                    extracted_params.setdefault("turnback", line_info["turnback"])
                    extracted_params.setdefault("cycle_time", line_info["cycle_time"])
                    extracted_params.setdefault("h_min", line_info["min_headway"])
                    extracted_params.setdefault("h_max", line_info["max_headway"])
                    extracted_params.setdefault("C", line_info["capacity"])

            first_line = list(self.network['lines'].values())[0]
            if 'h_min' not in extracted_params:
                extracted_params['h_min'] = first_line.get('min_headway', 120)
            if 'h_max' not in extracted_params:
                extracted_params['h_max'] = first_line.get('max_headway', 600)
            if 'C' not in extracted_params:
                extracted_params['C'] = first_line.get('train_capacity', 1500)
            if 'N' not in extracted_params:
                extracted_params['N'] = 30

            template = self.model_library.get_model(problem_type_str)
            if not template:
                template = self.model_library.get_model(ProblemType.HEADWAY.value)

            # 💥 消融开关：如果关闭 adapt，直接跳过自适应注入
            if self.ablation_mode == 'no_adapt':
                adaptation_trace = []
                print("   ⚠️ [Ablation] Model Adaptation 机制已禁用")
            else:
                adaptation_result = self.model_adaptation_agent.adapt(problem_type_str, query, template)
                template = adaptation_result["adapted_model"]
                adaptation_trace = adaptation_result["adaptation_trace"]

            math_model = self.math_agent.generate_model(user_query, extracted_params, template)
            math_model = self.model_linearizer.linearize(math_model)
            print("   📐 数学模型已生成并线性化（用于报告）。")

            spec = self.spec_builder.build(problem_type_str, extracted_params)
            code = self.code_agent.generate_code(spec, math_model)
            
            with open("generated_gurobi.py", "w", encoding="utf-8") as f:
                f.write(code)

            print("⚙️ Debug Agent 开始执行...")
            result = self.debug_agent.execute_and_debug(code, user_query)
            result["adaptation_trace"] = adaptation_trace

            if "variables" in result and isinstance(result["variables"], dict):
                result.update(result["variables"])

            if "frequency" in result and "headway" not in result and result["frequency"] > 0:
                result["headway"] = 3600 / result["frequency"]

            opt_status = result.get('status')
            is_optimal = (opt_status in [2, 'Optimal', 'OPTIMAL']) or (opt_status == GRB.OPTIMAL)

            if is_optimal:
                answer = None
                vars_dict = result.get('variables', {})
                if not isinstance(vars_dict, dict):
                    vars_dict = {}

                if problem_type_str == "capacity":
                    answer = vars_dict.get('frequency') or vars_dict.get('N')
                    if answer is None and (vars_dict.get('headway') or vars_dict.get('h')):
                        h_val = vars_dict.get('headway') or vars_dict.get('h')
                        if h_val > 0: answer = 3600 / h_val
                
                elif problem_type_str in ["rolling_stock", "crew", "passenger_assignment", "network_flow"]:
                    answer = result.get('objective_value') or result.get('objective')
                    if answer is None: 
                        answer = "Optimal"
                
                elif problem_type_str == "regenerative":
                    answer = vars_dict.get('utilization') or result.get('objective_value')
                
                elif problem_type_str in ["delay", "rescheduling"]:
                    for key, val in vars_dict.items():
                        if key.startswith('D') and isinstance(val, list) and len(val) > 0:
                            answer = val[-1]
                            break
                        elif key in ['D_N', 'D_n', 'D', 'Net_Delay']:
                            answer = val if not isinstance(val, list) else val[-1]
                            break
                    if answer is None:
                        answer = result.get('objective_value') or result.get('objective')
                
                else:
                    answer = vars_dict.get('headway') or vars_dict.get('h') or vars_dict.get('optimal_h')
                    if answer is None and vars_dict.get('frequency'):
                        freq = vars_dict.get('frequency')
                        if freq > 0: answer = 3600 / freq

                if answer is None and problem_type_str not in ['headway', 'delay', 'capacity']:
                    answer = result.get('objective_value') or result.get('objective')

                if answer is None:
                    print("⚠️ 求解状态为 Optimal 但未找到合适的 answer 字段")
                else:
                    result['extracted_answer'] = answer
                    if problem_type_str == "headway" and "headway" not in result and "h" not in result:
                        result["headway"] = answer

            elif opt_status in ['unbounded', 'Unbounded', 'UNBOUNDED', 5, '5']:
                answer = "Unbounded"
                print("⚠️ 求解器判定为无界 (Unbounded)，系统发散")
            else:
                answer = "Infeasible"
                print("⚠️ 求解失败，模型返回状态：", opt_status)

            result["problem_type"] = problem_type_str

            constraint_check = {"headway_constraint": False, "capacity_constraint": True}
            if is_optimal and answer is not None and answer != "Infeasible":
                is_valid, reason = self.headway_verifier.verify_headway(result, extracted_params)
                constraint_check["headway_constraint"] = is_valid
                warnings_list = self.solution_verifier.verify(problem_type_str, extracted_params, result)
                result['warnings'] = warnings_list
            else:
                result['warnings'] = []

            stats = SolverStatisticsCollector.collect(result)
            result.update(stats)

            recommendations = []
            scenario_results = []
            sensitivity_results = []
            operation_plans = {}
            pareto_front = []
            if is_optimal and answer is not None and answer != "Infeasible":
                recommendations = self.recommendation_agent.generate_recommendations(problem_type_str, extracted_params, result)
                scenario_results = self.scenario_evaluator.evaluate(problem_type_str, extracted_params, recommendations)
                if problem_type_str == "headway":
                    sensitivity_results = self.sensitivity_analyzer.analyze_headway(extracted_params)
                    operation_plans = self.operation_plan_agent.build_plan(sensitivity_results)
                    pareto_front = self.pareto_analyzer.analyze(sensitivity_results)

            reoptimization_results = []
            impact_results = []
            if is_optimal and answer is not None and answer != "Infeasible" and recommendations:
                scenarios = self.scenario_generator.generate(recommendations)
                for scenario in scenarios:
                    scenario_result = self.reoptimization_agent.resolve_headway(extracted_params, scenario["headway"])
                    reoptimization_results.append(scenario_result)
                    impact_results.append(self.impact_analyzer.compare(result, scenario_result))

            model_selection = self.model_explainability_agent.generate_model_selection(problem_type_str, template)
            constraint_explanation = self.model_explainability_agent.generate_constraint_explanation(spec)
            solver_explanation = self.model_explainability_agent.generate_solver_explanation(result)
            assumption_explanation = self.model_explainability_agent.generate_assumption_explanation(extracted_params)
            operational_recommendation = self.model_explainability_agent.generate_operational_recommendation(problem_type_str, result, extracted_params)
            kpi_analysis = self.model_explainability_agent.generate_kpi_analysis(problem_type_str, result, extracted_params)
            shadow_price = self.model_explainability_agent.generate_shadow_price_explanation(result)
            sensitivity = self.model_explainability_agent.generate_sensitivity_analysis(extracted_params, result)

            report = self.report_gen.generate_report(
                user_query, math_model, result,
                model_selection, constraint_explanation, solver_explanation,
                assumption_explanation, operational_recommendation,
                kpi_analysis, shadow_price, sensitivity,
                recommendations, scenario_results, sensitivity_results,
                operation_plans, pareto_front, reoptimization_results, impact_results
            )

            if operational_recommendation:
                explanation_short = " ".join(operational_recommendation)
            elif answer is not None and answer != "Infeasible":
                explanation_short = f"最优发车间隔为 {answer} 秒。"
            else:
                explanation_short = "求解失败，无法获得可行解。"

            reasoning = [
                f"问题类型: {problem_type_str}",
                f"意图: {intent}",
                f"参数完整性: {feasibility['completeness_score']}%",
                f"求解器状态: {opt_status}"
            ]
            if adaptation_trace:
                reasoning.append(f"模型适应: {adaptation_trace}")

            trace = ["参数提取", "可行性检查", "模型适应", "代码生成", "求解执行", "结果校验"]

            variables = {k: extracted_params.get(k) for k in ['Q', 'C', 'N', 'h_min', 'h_max'] if extracted_params.get(k) is not None}
            if answer is not None and answer != "Infeasible":
                variables['headway'] = answer

            factors = list(variables.keys())

            mapping_prompt = f"""
你是一个运筹学图理论专家。请分析以下数学模型，提取出"约束（Constraints）"和"决策变量（Variables）"之间的二分图映射关系。
必须严格以 JSON 格式输出，键为约束的英文特征名，值为该约束公式中包含的决策变量列表。
示例：{{"DelayPropagation": ["D_k", "D_k+1"], "CapacityLimit": ["h", "N"]}}

当前数学模型：
{math_model}
"""
            constraint_mapping = self.api.query_json(mapping_prompt, system_prompt="只输出纯 JSON，不要任何 Markdown 标记或多余文字。")
            if not isinstance(constraint_mapping, dict):
                constraint_mapping = {}

            wrapped = self.wrap_output(
                answer=answer,
                explanation=explanation_short,
                reasoning=reasoning,
                trace=trace,
                variables=variables,
                factors=factors,
                constraint_check=constraint_check,
                report_string=report,
                is_valid=(is_optimal or answer == "Unbounded") and (answer is not None and answer != "Infeasible"),
                constraint_mapping=constraint_mapping
            )

            elapsed = (datetime.now() - start).total_seconds()
            self.history.append({'query': user_query, 'result': result, 'params': extracted_params, 'time': elapsed})
            print(f"✅ 处理完成，耗时 {elapsed:.1f} 秒")
            return wrapped

        except Exception as e:
            error_msg = f"求解失败: {str(e)}"
            import traceback
            traceback.print_exc()
            return self.wrap_output(
                answer="Infeasible",
                explanation=error_msg,
                reasoning=["系统异常捕获"],
                trace=[],
                variables={},
                factors=[],
                constraint_check={"system": False},
                report_string=f"系统无法处理该请求，原因：{error_msg}",
                is_valid=False,
                constraint_mapping={}
            )

# ==================== 交互式运行函数 ====================
def run_optimization_system():
    print("="*80)
    print("     🧠 香港地铁调度优化系统 v21.0 (多模型智能交互版)")
    print("="*80)
    
    # 修改这里的提示语
    api_key_input = input("请输入 API 密钥 (留空则尝试读取系统环境变量): ").strip()
    
    # 🤖 智能路由：根据密钥前缀自动判定提供商
    current_provider = "qwen" # 默认值
    
    if api_key_input:
        if api_key_input.startswith("sk-jb-"):
            current_provider = "gpt-5.2"
            print("💡 系统检测到 JBridge 密钥，已自动切换至 [GPT-4o] 引擎")
        elif api_key_input.startswith("sk-8f"): 
            current_provider = "deepseek"
            print("💡 系统检测到 DeepSeek 密钥，已自动切换至 [DeepSeek] 引擎")
        elif api_key_input.startswith("sk-ws-"):
            current_provider = "qwen"
            print("💡 系统检测到阿里云密钥，已自动切换至 [Qwen] 引擎")
        else:
            print("\n未识别的密钥格式，请选择对应的模型提供商:")
            print("[1] Qwen (通义千问)")
            print("[2] DeepSeek")
            print("[3] GPT-5.2 (中转)")
            choice = input("请选择 (默认1): ").strip()
            if choice == '2': current_provider = "deepseek"
            elif choice == '3': current_provider = "gpt-5.2"

    # 初始化 API 实例
    try:
        if not api_key_input:
            api = DeepSeekAPI(provider="qwen") # 留空时默认尝试用 Qwen 的环境变量
        else:
            api = DeepSeekAPI(api_key=api_key_input, provider=current_provider)
    except ValueError as e:
        print(f"\n{e}\n程序已退出。")
        return
          
    # 将创建好的完整 api 对象实例注入给 orchestrator
    orchestrator = Orchestrator(api_client=api)

    print("\n💡 示例问题：")
    print("1. 荃湾线客流8500人/小时，列车容量1500人，求最优发车间隔")
    print("2. 港岛线高峰时段如何调整发车间隔以最小化乘客等待时间？")
    print("3. 如何优化车底交路以最小化列车数量？")
    print("4. 晚点5分钟，如何重调度？")
    print("5. 金钟站早高峰拥挤，如何改善？")

    try:
        from google.colab import files
        COLAB = True
    except:
        COLAB = False

    while True:
        print("\n" + "-"*80)
        query = input("请输入您的地铁优化问题 (输入'退出'结束): ").strip()
        if query.lower() in ['退出', 'exit', 'quit', 'q']:
            break
        if not query:
            continue

        response = orchestrator.process(query)
        if isinstance(response, dict) and 'report' in response:
            print("\n" + response['report'])
        else:
            print("\n" + str(response))

        while True:
            opt = input("\n操作: [1]保存报告 [2]查看代码 [3]继续提问 [4]退出 : ").strip()
            if opt == '1' and isinstance(response, dict) and 'report' in response:
                fname = f"metro_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(response['report'])
                print(f"✅ 报告已保存: {fname}")
                if COLAB:
                    files.download(fname)
            elif opt == '2' and isinstance(response, dict) and 'code' in response:
                print("\n📜 完整求解代码:\n" + response['code'])
            elif opt == '3':
                break
            elif opt == '4':
                return
            else:
                print("无效选项")

    if orchestrator.history:
        print("\n📊 本次会话统计")
        print(f"处理问题数: {len(orchestrator.history)}")
        avg_t = sum(h['time'] for h in orchestrator.history) / len(orchestrator.history)
        print(f"平均耗时: {avg_t:.1f} 秒")

if __name__ == "__main__":
    run_optimization_system()