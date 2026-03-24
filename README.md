# Financial Risk Agent: 基于多模态 RAG 的智能财务风控评审引擎

[Image of a system architecture diagram for a multimodal RAG financial analysis agent, showing PDF ingestion, 3-tier metadata routing, Vision LLM parsing, and Streamlit UI]

## 项目背景 (Background)
在真实的商业分析场景中（如处理美团等大型科技公司的非标、数百页年报），传统的纯文本 RAG (Retrieval-Augmented Generation) 方案往往面临两个致命痛点：
1. **目录迷失 (TOC Disconnect)**：复杂财报的真实财务数据表通常深藏在数十页的业务回顾之后，常规的前置截断检索极易失效。
2. **空间失真 (Spatial Flattening)**：传统 OCR 会将二维财务表格拉平为一维乱序字符串，导致大模型因严格的防幻觉机制而无法准确提取财务科目。

本项目旨在构建一个**具备自主规划能力 (Agentic Workflow) 的自动化风控评审引擎**。通过引入多模态视觉大模型和多层降级路由机制，彻底解决复杂长文本金融文档的自动化解析难题。

## 核心架构与亮点 (Key Features)

### 1. 三层工业级高可用路由 (3-Tier Graceful Degradation Routing)
摒弃低效的全量 OCR，系统设计了逐级降级的智能寻址策略，将核心报表定位时间从数分钟降至 **0.1 秒**：
* **最佳路径 (Metadata Bypass)**：利用 `pypdf` 递归提取 PDF 原生大纲书签，实现光速精准定位。
* **次优路径 (Vision TOC Radar)**：若无书签，退化为轻量级视觉特征提取，引导 LLM 在前 15 页动态寻找目录。
* **兜底路径 (Human-in-the-loop)**：提供可视化界面的“专家模式”，允许分析师手动注入黄金页码，保证系统 100% 可用。

### 2. 多模态表格降维打击 (Multimodal Table Parsing)
针对双语言、非标排版的港股财报，引入支持 Vision 接口的多模态模型（Claude 3.5 Sonnet）。系统将 PDF 目标页转化为高清图像，利用视觉大模型的**空间感知能力**进行数据提取，完美绕过纯文本解析时的行列错位问题。

### 3. 动态风控指标计算 (Dynamic Risk Metrics)
内置财务风控引擎，动态提取“收入”、“年内溢利”、“流动资产”等核心科目，自动计算并输出结构化 JSON 数据：
* 净利率 (Net Profit Margin)
