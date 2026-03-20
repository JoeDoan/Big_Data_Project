import streamlit as st
import os
import time

# Import our working agent loops
from agent import run_lexguard_agent as run_baseline_agent
from adapted_agent import run_adapted_agent
from monitor import MetricsCollector

# ═══════════════════════════════════════════════
# 1. Page Configuration & Custom CSS
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="LexGuard Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme CSS with glassmorphism and animations
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .sub-header {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
    }

    /* Metric badges */
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-purple { background: rgba(124, 58, 237, 0.2); color: #A78BFA; }
    .badge-blue   { background: rgba(59, 130, 246, 0.2); color: #93C5FD; }
    .badge-green  { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; }
    .badge-amber  { background: rgba(245, 158, 11, 0.2); color: #FCD34D; }
    .badge-red    { background: rgba(239, 68, 68, 0.2);  color: #FCA5A5; }

    /* Latency tag */
    .latency-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        margin-left: 8px;
    }

    /* Risk badges */
    .risk-high   { background: rgba(239, 68, 68, 0.15); color: #FCA5A5; padding: 2px 10px; border-radius: 6px; font-weight: 600; }
    .risk-medium { background: rgba(245, 158, 11, 0.15); color: #FCD34D; padding: 2px 10px; border-radius: 6px; font-weight: 600; }
    .risk-low    { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; padding: 2px 10px; border-radius: 6px; font-weight: 600; }

    /* Trace log styling */
    .trace-step {
        padding: 6px 12px;
        border-left: 3px solid #7C3AED;
        margin: 6px 0;
        font-size: 0.8rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 0 6px 6px 0;
    }

    /* Query history items */
    .history-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        font-size: 0.8rem;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.2s ease;
        color: #CBD5E1;
    }
    .history-item:hover {
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.3);
    }

    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online  { background: #10B981; box-shadow: 0 0 6px #10B981; }
    .status-offline { background: #EF4444; box-shadow: 0 0 6px #EF4444; }
    .status-unknown { background: #F59E0B; box-shadow: 0 0 6px #F59E0B; }

    /* Sidebar section titles */
    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748B;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* Streamlit expander override */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Animated gradient border effect for chat input */
    .stChatInput > div {
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 12px !important;
    }
    .stChatInput > div:focus-within {
        border-color: #7C3AED !important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 2. Session State Initialization
# ═══════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am **LexGuard** ⚖️. What contract clauses would you like me to audit today?",
         "trace": None, "latency": None, "risk": None}
    ]

if "collector" not in st.session_state:
    st.session_state.collector = MetricsCollector()

if "query_history" not in st.session_state:
    st.session_state.query_history = []


# ═══════════════════════════════════════════════
# 3. Sidebar
# ═══════════════════════════════════════════════
with st.sidebar:
    # ── Pipeline Selector ──
    st.markdown('<div class="sidebar-title">⚙️ Pipeline</div>', unsafe_allow_html=True)
    pipeline_choice = st.radio(
        "Select audit pipeline:",
        ("Baseline (Gemini API)", "Adapted (Mistral PEFT)"),
        label_visibility="collapsed"
    )

    if pipeline_choice == "Adapted (Mistral PEFT)":
        st.markdown("""<div class="glass-card">
            <span class="metric-badge badge-purple">PEFT</span>
            Domain-adapted LoRA model + local RAG retrieval + deterministic risk rules
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="glass-card">
            <span class="metric-badge badge-blue">BASELINE</span>
            Gemini 2.5 Flash + Snowflake retrieval + function calling
        </div>""", unsafe_allow_html=True)

    # ── System Status ──
    st.markdown('<div class="sidebar-title">📡 System Status</div>', unsafe_allow_html=True)

    gemini_status = "online" if os.getenv("GEMINI_API_KEY") else "offline"
    snow_status = "online" if os.getenv("SNOW_ACCOUNT") else "offline"
    colab_status = "unknown" if os.getenv("COLAB_API_URL") else "offline"

    st.markdown(f"""<div class="glass-card">
        <div><span class="status-dot status-{gemini_status}"></span> Gemini API</div>
        <div><span class="status-dot status-{snow_status}"></span> Snowflake DB</div>
        <div><span class="status-dot status-{colab_status}"></span> Colab PEFT Server</div>
    </div>""", unsafe_allow_html=True)

    # ── Analytics Dashboard ──
    collector = st.session_state.collector
    if collector.total_queries() > 0:
        st.markdown('<div class="sidebar-title">📊 Session Analytics</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Queries", collector.total_queries())
        col2.metric("Avg Latency", f"{collector.avg_latency()}s")

        col3, col4 = st.columns(2)
        col3.metric("Success Rate", f"{collector.success_rate()}%")

        # Pipeline breakdown
        breakdown = collector.pipeline_breakdown()
        if len(breakdown) > 1:
            st.markdown("**Pipeline Usage:**")
            for pipeline, count in breakdown.items():
                short_name = pipeline.split("(")[0].strip()
                st.progress(count / collector.total_queries(), text=f"{short_name}: {count}")

        # Average latency by pipeline
        avg_by_pipeline = collector.avg_latency_by_pipeline()
        if avg_by_pipeline:
            st.markdown("**Avg Latency by Pipeline:**")
            for pipeline, avg in avg_by_pipeline.items():
                short_name = pipeline.split("(")[0].strip()
                st.markdown(f"<span class='metric-badge badge-blue'>{short_name}</span> {avg}s", unsafe_allow_html=True)

        # Tool usage
        tool_usage = collector.tool_usage_breakdown()
        if tool_usage:
            st.markdown("**Tool Calls:**")
            for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
                st.markdown(f"<span class='metric-badge badge-green'>{tool}</span> ×{count}", unsafe_allow_html=True)

    # ── Query History ──
    if st.session_state.query_history:
        st.markdown('<div class="sidebar-title">🕐 Query History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            risk_badge = ""
            if item.get("risk") == "High":
                risk_badge = '<span class="risk-high">HIGH</span>'
            elif item.get("risk") == "Medium":
                risk_badge = '<span class="risk-medium">MED</span>'
            elif item.get("risk") == "Low":
                risk_badge = '<span class="risk-low">LOW</span>'

            st.markdown(f"""<div class="history-item">
                <div>{item['query'][:50]}{'...' if len(item['query']) > 50 else ''}</div>
                <div style="margin-top:4px">
                    <span class="latency-tag">{item.get('latency', '?')}s</span>
                    {risk_badge}
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 4. Main Chat Area
# ═══════════════════════════════════════════════
st.markdown('<h1 class="main-header">⚖️ LexGuard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Neuro-Symbolic Compliance Auditor for Contract Risk Analysis</p>', unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show latency badge on assistant messages
        if msg["role"] == "assistant" and msg.get("latency") is not None:
            risk = msg.get("risk", "N/A")
            risk_class = "risk-high" if risk == "High" else "risk-medium" if risk == "Medium" else "risk-low" if risk == "Low" else "badge-blue"
            st.markdown(f"""
                <span class="latency-tag">⏱ {msg['latency']}s</span>
                <span class="{risk_class}" style="font-size:0.75rem">{risk} Risk</span>
            """, unsafe_allow_html=True)

        # Show debug trace in expandable panel
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔍 Execution Trace & Debug Log"):
                for step in msg["trace"]:
                    step_type = step.get("step", "")
                    step_time = step.get("time", "")
                    time_str = f" — {step_time}s" if step_time else ""

                    if step_type == "start":
                        st.markdown(f'<div class="trace-step">📝 <b>Query Received:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "tool_call":
                        tool = step.get("tool", "unknown")
                        preview = step.get("result_preview", "")
                        st.markdown(f'<div class="trace-step">🛠️ <b>Tool Call:</b> <code>{tool}</code>{time_str}<br><small style="color:#94A3B8">{preview}</small></div>', unsafe_allow_html=True)
                    elif step_type == "model_inference":
                        st.markdown(f'<div class="trace-step">🤖 <b>Model Inference:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "response":
                        st.markdown(f'<div class="trace-step">✅ <b>Response Generated</b>{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "error":
                        st.markdown(f'<div class="trace-step" style="border-color:#EF4444">❌ <b>Error:</b> {step.get("detail", "")}</div>', unsafe_allow_html=True)
                    elif step_type == "greeting_filter":
                        st.markdown(f'<div class="trace-step">👋 <b>Greeting Detected</b> — skipped pipeline</div>', unsafe_allow_html=True)
                    elif step_type == "no_results":
                        st.markdown(f'<div class="trace-step" style="border-color:#F59E0B">⚠️ <b>{step.get("detail", "")}</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="trace-step">ℹ️ {step.get("detail", step_type)}{time_str}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 5. User Chat Input & Agent Execution
# ═══════════════════════════════════════════════
if prompt := st.chat_input("e.g., Are there any high-risk indemnification clauses?"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "trace": None, "latency": None, "risk": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Start metrics collection
    collector = st.session_state.collector
    pipeline_label = pipeline_choice
    metrics = collector.start(prompt, pipeline_label)

    # Execute agent with error handling
    with st.chat_message("assistant"):
        with st.spinner(f"🔍 Analyzing with **{pipeline_choice}**..."):
            try:
                if pipeline_choice == "Adapted (Mistral PEFT)":
                    result = run_adapted_agent(prompt)
                else:
                    result = run_baseline_agent(prompt)

                response_text = result["response"]
                trace = result["trace"]
                tool_calls = result["tool_calls"]
                retrieval_count = result["retrieval_count"]
                risk_level = result["risk_level"]
                success = result["success"]

            except Exception as e:
                response_text = f"⚠️ **An error occurred:** {str(e)}\n\nPlease check your API keys and network connection, then try again."
                trace = [{"step": "error", "detail": str(e), "time": 0}]
                tool_calls = []
                retrieval_count = 0
                risk_level = "N/A"
                success = False

        # Finalize metrics
        collector.finish(metrics, success=success, tool_calls=tool_calls,
                        retrieval_count=retrieval_count, risk_level=risk_level)

        # Display response
        st.markdown(response_text)

        # Inline latency + risk badge
        latency = metrics.latency_s
        risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low" if risk_level == "Low" else "badge-blue"
        st.markdown(f"""
            <span class="latency-tag">⏱ {latency}s</span>
            <span class="{risk_class}" style="font-size:0.75rem">{risk_level} Risk</span>
        """, unsafe_allow_html=True)

        # Debug trace expander
        if trace:
            with st.expander("🔍 Execution Trace & Debug Log"):
                for step in trace:
                    step_type = step.get("step", "")
                    step_time = step.get("time", "")
                    time_str = f" — {step_time}s" if step_time else ""

                    if step_type == "start":
                        st.markdown(f'<div class="trace-step">📝 <b>Query Received:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "tool_call":
                        tool = step.get("tool", "unknown")
                        preview = step.get("result_preview", "")
                        st.markdown(f'<div class="trace-step">🛠️ <b>Tool Call:</b> <code>{tool}</code>{time_str}<br><small style="color:#94A3B8">{preview}</small></div>', unsafe_allow_html=True)
                    elif step_type == "model_inference":
                        st.markdown(f'<div class="trace-step">🤖 <b>Model Inference:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "response":
                        st.markdown(f'<div class="trace-step">✅ <b>Response Generated</b>{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "error":
                        st.markdown(f'<div class="trace-step" style="border-color:#EF4444">❌ <b>Error:</b> {step.get("detail", "")}</div>', unsafe_allow_html=True)
                    elif step_type == "greeting_filter":
                        st.markdown(f'<div class="trace-step">👋 <b>Greeting Detected</b> — skipped pipeline</div>', unsafe_allow_html=True)
                    elif step_type == "no_results":
                        st.markdown(f'<div class="trace-step" style="border-color:#F59E0B">⚠️ <b>{step.get("detail", "")}</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="trace-step">ℹ️ {step.get("detail", step_type)}{time_str}</div>', unsafe_allow_html=True)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "trace": trace,
        "latency": latency,
        "risk": risk_level
    })

    # Add to query history
    st.session_state.query_history.append({
        "query": prompt,
        "latency": latency,
        "risk": risk_level,
        "pipeline": pipeline_choice
    })

    st.rerun()