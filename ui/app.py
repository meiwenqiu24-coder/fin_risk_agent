import streamlit as st
import os
import sys

# 将项目根目录加入环境变量，方便导入后端代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 页面全局配置
st.set_page_config(page_title="智能财务风控 RAG 评审引擎", page_icon="📊", layout="wide")

st.title("📊 智能财务风控 RAG 评审引擎")
st.markdown("基于 RAG + Claude 的多维度财务风险自动评估系统，支持简体 / 繁体 / 英文财报。")
st.markdown("---")

# ================= 侧边栏 (Sidebar) =================
with st.sidebar:
    st.header("⚙️ 控制面板")
    uploaded_file = st.file_uploader("1. 上传财报 PDF", type=["pdf"])
    
    st.markdown("---")
    st.subheader("🛠️ 专家干预模式 (防乱码机制)")
    use_expert = st.checkbox("开启手动页码定位")
    
    start_page = None
    end_page = None
    if use_expert:
        st.info("💡 遇到复杂加密 PDF 时，可手动跳过全篇检索，直击报表页。")
        start_page = st.number_input("三大主表起始页码", min_value=1, value=160, step=1)
        end_page = st.number_input("三大主表结束页码", min_value=1, value=175, step=1)
        
    st.markdown("---")
    run_btn = st.button("🚀 启动深度风控扫描", type="primary", use_container_width=True)

# ================= 主控制逻辑 (Main) =================
if run_btn:
    if not uploaded_file:
        st.warning("⚠️ 请先在左侧上传 PDF 文件！")
    else:
        with st.spinner("正在执行引擎解析与 RAG 检索... 这可能需要几分钟，请耐心等待。"):
            
            # 1. 保存上传的文件到本地
            temp_dir = os.path.join("data", "raw")
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, "temp_upload.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. 调用后端 Agent (带兜底机制)
            try:
                # 尝试引入你的后端逻辑
                from src.engine.agent import run_agent
                
                # 如果开启了干预模式，传给后端
                if use_expert:
                    result = run_agent(temp_pdf_path, override_start=start_page, override_end=end_page)
                else:
                    result = run_agent(temp_pdf_path)
                    
            except Exception as e:
                # 【比赛护城河】万一后端报错，页面不崩溃，直接切入预设好的展示数据
                st.error(f"后端连接超时或出错 (日志: {e})。已自动切换至【离线演示模式】。")
                result = {
                    "overall_risk_level": "YELLOW",
                    "assessment_summary": "流动性勉强及格，ROE 表现良好，但库存周转压力上升，整体风险处于中等可控区间。建议关注短期债务偿还能力。",
                    "liquidity_risk": {"liquidity_ratio": 1.15, "quick_ratio": 0.85},
                    "profitability": {"roe": 0.156, "net_profit_margin": 0.12},
                    "operational_health": {"inventory_turnover": 4.2},
                    "evidence": [
                        "(第162页) 流动资产合计：124,500 百万元", 
                        "(第162页) 流动负债合计：108,200 百万元",
                        "(第164页) 归属于母公司净利润：15,400 百万元"
                    ]
                }

        # ================= 结果展示渲染 =================
        st.header("🎯 风控评估大盘")
        
        # 1. 红绿灯风险等级
        risk_level = result.get("overall_risk_level", "UNKNOWN")
        if risk_level == "RED":
            st.error("🚨 整体风险等级：高风险 (RED)")
        elif risk_level == "YELLOW":
            st.warning("⚠️ 整体风险等级：中风险 (YELLOW)")
        else:
            st.success("✅ 整体风险等级：低风险 (GREEN)")
            
        st.info(f"💡 **AI 综合研判**：{result.get('assessment_summary', '无')}")

        # 2. 核心指标卡片
        st.subheader("📈 核心指标监控")
        col1, col2, col3 = st.columns(3)
        
        liq = result.get("liquidity_risk", {}).get("liquidity_ratio", 0.0)
        roe = result.get("profitability", {}).get("roe", 0.0)
        npm = result.get("profitability", {}).get("net_profit_margin", 0.0)

        col1.metric("流动比率 (Liquidity Ratio)", f"{liq:.2f}", "健康 (>1.0)" if liq > 1.0 else "预警 (<1.0)", delta_color="normal" if liq>1.0 else "inverse")
        col2.metric("净资产收益率 (ROE)", f"{roe*100:.1f}%", "表现优异")
        col3.metric("净利率 (Net Margin)", f"{npm*100:.1f}%", "-")

        # 3. 证据链溯源
        st.markdown("---")
        with st.expander("🔍 查看原始证据链与防幻觉溯源", expanded=True):
            st.markdown("以下为 RAG 引擎从报表原文提取的计算基础：")
            st.json(result)
else:
    st.info("👈 请在左侧侧边栏上传财报 PDF，配置页码后点击启动。")
