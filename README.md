#  Financial Risk Agent: 基于多模态 RAG 的智能财务风控评审引擎



##  项目背景 (Background)
在真实的商业分析场景中（如处理美团等大型科技公司的非标、数百页年报），传统的纯文本 RAG (Retrieval-Augmented Generation) 方案往往面临两个致命痛点：
1. **目录迷失 (TOC Disconnect)**：复杂财报的真实财务数据表通常深藏在数十页的业务回顾之后，常规的前置截断检索极易失效。
2. **空间失真 (Spatial Flattening)**：传统 OCR 会将二维财务表格拉平为一维乱序字符串，导致大模型因严格的防幻觉机制而无法准确提取财务科目。

本项目旨在构建一个**具备自主规划能力 (Agentic Workflow) 的自动化风控评审引擎**。通过引入多模态视觉大模型和多层降级路由机制，彻底解决复杂长文本金融文档的自动化解析难题。

##  核心架构与亮点 (Key Features)

### 1.  三层工业级高可用路由 (3-Tier Graceful Degradation Routing)
摒弃低效的全量 OCR，系统设计了逐级降级的智能寻址策略，将核心报表定位时间从数分钟降至 **0.1 秒**：
* **最佳路径 (Metadata Bypass)**：利用 `pypdf` 递归提取 PDF 原生大纲书签，实现光速精准定位。
* **次优路径 (Vision TOC Radar)**：若无书签，退化为轻量级视觉特征提取，引导 LLM 在前 15 页动态寻找目录。
* **兜底路径 (Human-in-the-loop)**：提供可视化界面的“专家模式”，允许分析师手动注入黄金页码，保证系统 100% 可用。

### 2.  多模态表格降维打击 (Multimodal Table Parsing)
针对双语言、非标排版的港股财报，引入支持 Vision 接口的多模态模型（Claude 3.5 Sonnet）。系统将 PDF 目标页转化为高清图像，利用视觉大模型的**空间感知能力**进行数据提取，完美绕过纯文本解析时的行列错位问题。

### 3.  动态风控指标计算 (Dynamic Risk Metrics)
内置财务风控引擎，动态提取“收入”、“年内溢利”、“流动资产”等核心科目，自动计算并输出结构化 JSON 数据：
* 净利率 (Net Profit Margin)
* 流动比率 (Liquidity Ratio)
* 净资产收益率 (ROE)
*(系统内置了平滑替换逻辑，适应不同上市地的财务披露规范差异)*

### 4.  可解释性交互大盘 (Explainable AI Dashboard)
基于 Streamlit 搭建了端到端的数据分析交互界面。提供红黄绿动态风险预警，并内置**“原始证据链溯源”**折叠面板，确保 AI 生成的每一条数据都可审计、防幻觉。

##  技术栈 (Tech Stack)
* **核心框架:** Python, LangChain
* **多模态大模型:** Anthropic Claude 3.5 Sonnet (Vision)
* **文档智能:** pypdf (Metadata Extraction), pdf2image, EasyOCR
* **前端展示:** Streamlit

##  快速启动 (Quick Start)

### 1. 克隆仓库与环境配置
```bash
git clone [https://github.com/meiwenqiu24-coder/fin_risk_agent.git](https://github.com/meiwenqiu24-coder/fin_risk_agent.git)
cd fin_risk_agent

# 推荐使用 conda 管理虚拟环境
conda create -n risk_agent python=3.10
conda activate risk_agent

# 安装依赖
pip install -r requirements.txt
