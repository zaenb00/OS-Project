import sys
import os
import copy
import psutil
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

# -------------------------------------------------
# Path setup
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Project imports
# -------------------------------------------------
from src.utils.workload import generate_workload
from src.utils.metrics import (
    average_waiting_time,
    average_turnaround_time,
    cpu_utilization,
)
from src.schedulers.fcfs import fcfs_scheduler
from src.schedulers.sjf import sjf_scheduler
from src.schedulers.round_robin import round_robin_scheduler
from src.schedulers.rl_schedular import rl_dqn_scheduler
from src.utils.realtime_cpu import get_realtime_workload
from src.env.cpu_env import CPUSchedulingEnv
from src.agents.dqn import DQNAgent

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(page_title="AI CPU Scheduler", layout="wide")
st.title("🧠 AI-Powered CPU Scheduling Simulator")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "cpu_usage" not in st.session_state:
    st.session_state.cpu_usage = []

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

num_cores = st.sidebar.slider("Number of CPU Cores", 1, 8, 1)
num_processes = st.sidebar.slider("Number of Processes", 3, 20, 10)

scheduler_choice = st.sidebar.selectbox(
    "Select Scheduler",
    [
        "FCFS",
        "SJF",
        "Round Robin",
        "RL Scheduler (DQN)",
        "RL Scheduler (Online)",
    ],
)

quantum = 2
if scheduler_choice == "Round Robin":
    quantum = st.sidebar.slider("Time Quantum", 1, 5, 2)

seed = st.sidebar.number_input("Random Seed", value=42)

# -------------------------------------------------
# Run scheduler
# -------------------------------------------------
completed = []
total_time = 0
idle_time = 0

if scheduler_choice in ["FCFS", "SJF", "Round Robin", "RL Scheduler (DQN)"]:
    workload = generate_workload(num_processes=num_processes, seed=seed)

if scheduler_choice == "FCFS":
    completed, total_time, idle_time = fcfs_scheduler(
        copy.deepcopy(workload), num_cores=num_cores
    )

elif scheduler_choice == "SJF":
    completed, total_time, idle_time = sjf_scheduler(
        copy.deepcopy(workload), num_cores=num_cores
    )

elif scheduler_choice == "Round Robin":
    completed, total_time, idle_time = round_robin_scheduler(
        copy.deepcopy(workload), quantum=quantum, num_cores=num_cores
    )

elif scheduler_choice == "RL Scheduler (DQN)":
    completed, total_time, idle_time = rl_dqn_scheduler(
        copy.deepcopy(workload),
        num_cores=num_cores,
    )

elif scheduler_choice == "RL Scheduler (Online)":
    processes = get_realtime_workload()
    env = CPUSchedulingEnv(processes, num_cores=num_cores)

    state = env.reset()
    state_size = len(state)
    action_size = env.max_queue_size * num_cores

    # 🔁 Reset agent safely if shape changed
    if (
        st.session_state.agent is None
        or st.session_state.agent.state_size != state_size
        or st.session_state.agent.action_size != action_size
    ):
        st.session_state.agent = DQNAgent(state_size, action_size)

        # load pretrained if exists
        st.session_state.agent.load(
            "models/pretrained_rl_scheduler.pt",
            state_size=state_size,
            action_size=action_size,
            num_cores=num_cores,
        )

    agent = st.session_state.agent

    EPISODES_PER_RUN = 5

    for _ in range(EPISODES_PER_RUN):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                next_state, reward, done = env.step(None)
                action = None
            else:
                action = agent.act(state, valid_actions)
                next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()
        st.session_state.rewards.append(total_reward / max(1, env.time))

    os.makedirs("models", exist_ok=True)
    agent.save(
        "models/pretrained_rl_scheduler.pt",
        num_cores=num_cores,
    )

    completed = env.completed
    total_time = env.time
    idle_time = 0

# -------------------------------------------------
# NAVBAR TABS
# -------------------------------------------------
tabs = st.tabs([
    "🔴 Real-Time CPU",
    "🧠 RL Learning Curve",
    "📊 Results",
    "📋 Process Table",
    "🧩 Core Distribution",
    "📊 Gantt Chart",
    "📈 Scheduler Comparison",
    "🔥 Core Utilization Heatmap",
    "🧾 Process JSON",
])

# -------------------------------------------------
# TAB 1: Real-Time CPU
# -------------------------------------------------
with tabs[0]:
    st.subheader("🔴 Real-Time CPU Monitoring")
    cpu = psutil.cpu_percent(interval=0.5)
    st.session_state.cpu_usage.append(cpu)
    st.line_chart(pd.DataFrame({"CPU Usage (%)": st.session_state.cpu_usage}))
    st.write(f"**Active OS Processes:** {len(psutil.pids())}")

# -------------------------------------------------
# TAB 2: RL Learning Curve
# -------------------------------------------------
with tabs[1]:
    st.subheader("🧠 RL Learning Curve")
    if scheduler_choice == "RL Scheduler (Online)" and len(st.session_state.rewards) > 1:
        rewards = pd.Series(st.session_state.rewards)
        rolling = rewards.rolling(10).mean()
        st.line_chart(
            pd.DataFrame(
                {
                    "Reward": rewards,
                    "Rolling Avg (10)": rolling,
                }
            )
        )
    else:
        st.info("Run Online RL Scheduler to view learning curve.")

# -------------------------------------------------
# TAB 3: Results
# -------------------------------------------------
with tabs[2]:
    st.subheader("📊 Scheduling Metrics")
    avg_wait = average_waiting_time(completed)
    avg_turn = average_turnaround_time(completed)
    util = cpu_utilization(total_time, idle_time)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Waiting Time", f"{avg_wait:.2f}")
    c2.metric("Avg Turnaround Time", f"{avg_turn:.2f}")
    c3.metric("CPU Utilization (%)", f"{util:.2f}")

# -------------------------------------------------
# TAB 4: Process Table
# -------------------------------------------------
with tabs[3]:
    st.subheader("📋 Process Table")
    if completed: 
        st.dataframe(pd.DataFrame(completed)) 
    else: 
        st.info("No completed processes.")

# -------------------------------------------------
# TAB 5: Core Distribution
# -------------------------------------------------
with tabs[4]:
    st.subheader("🧩 Core-wise Process Distribution")
    core_map = {i: [] for i in range(num_cores)}
    for p in completed:
        if "core" in p:
            core_map[p["core"]].append(f"P{p['pid']}")

    rows = [
        {
            "Core": f"Core {k}",
            "Processes Executed": ", ".join(v) if v else "Idle",
        }
        for k, v in core_map.items()
    ]
    st.table(pd.DataFrame(rows))

# -------------------------------------------------
# Gantt chart renderer
# -------------------------------------------------
def render_gantt_chart(completed_processes):
    fig = go.Figure()

    for p in completed_processes:
        if p.get("start_time") is None or p.get("finish_time") is None:
            continue

        fig.add_bar(
            x=[p["finish_time"] - p["start_time"]],
            y=[f"Core {p.get('core', 0)}"],
            base=p["start_time"],
            orientation="h",
            name=f"P{p['pid']}",
        )

    fig.update_layout(
        title="CPU Scheduling Gantt Chart",
        xaxis_title="Time Units",
        yaxis_title="CPU Core",
        barmode="overlay",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 6: Gantt Chart
# -------------------------------------------------
with tabs[5]:
    st.subheader("CPU Scheduling Timeline (Gantt Chart)")
    render_gantt_chart(completed)

# -------------------------------------------------
# TAB 7: JSON
# -------------------------------------------------
with tabs[8]:
    st.subheader("🧾 Process Details (JSON)")
    st.json(json.loads(json.dumps(completed, default=str)))

def run_scheduler_for_comparison(name, workload, num_cores):
    if name == "FCFS":
        return fcfs_scheduler(copy.deepcopy(workload), num_cores)
    if name == "SJF":
        return sjf_scheduler(copy.deepcopy(workload), num_cores)
    if name == "Round Robin":
        return round_robin_scheduler(copy.deepcopy(workload), quantum=2, num_cores=num_cores)
    if name == "RL":
        return rl_dqn_scheduler(copy.deepcopy(workload), num_cores=num_cores)


with tabs[6]:
    st.subheader("📈 Scheduler Performance Comparison")
    st.caption("Comparison of classical schedulers vs RL-based scheduler")

    base_workload = generate_workload(
        num_processes=num_processes,
        seed=seed
    )

    rows = []

    for name in ["FCFS", "SJF", "Round Robin", "RL"]:
        comp, total, idle = run_scheduler_for_comparison(
            name, base_workload, num_cores
        )

        rows.append({
            "Scheduler": name,
            "Avg Waiting Time": round(average_waiting_time(comp), 2),
            "Avg Turnaround Time": round(average_turnaround_time(comp), 2),
            "CPU Utilization (%)": round(cpu_utilization(total, idle), 2),
            "Total Time": total,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.bar_chart(
        df.set_index("Scheduler")[["Avg Waiting Time", "Avg Turnaround Time"]]
    )


with tabs[7]:
    st.subheader("🔥 Per-Core Utilization Heatmap")
    st.caption("Visualizes load distribution across CPU cores")

    if not completed:
        st.info("Run a scheduler to generate utilization data.")
    else:
        core_usage = {i: 0 for i in range(num_cores)}

        for p in completed:
            if "core" in p and p.get("start_time") is not None:
                core_usage[p["core"]] += p["finish_time"] - p["start_time"]

        heatmap_df = pd.DataFrame({
            "Core": list(core_usage.keys()),
            "Busy Time": list(core_usage.values())
        })

        fig = go.Figure(
            data=go.Heatmap(
                z=[heatmap_df["Busy Time"]],
                x=heatmap_df["Core"],
                y=["CPU Utilization"],
                colorscale="Viridis",
            )
        )

        fig.update_layout(
            xaxis_title="CPU Core",
            yaxis_title="",
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)
