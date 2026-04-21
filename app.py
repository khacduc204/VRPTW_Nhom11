import streamlit as st
import pandas as pd
import os
import glob
from main import evaluate_algorithm, ALGORITHMS, _find_instance_file
from utils import set_use_numba
from data import load_solomon_instance

st.set_page_config(page_title="VRPTW Metaheuristics Web UI", layout="wide")

st.title("Chạy Thuật Toán VRPTW (GA, PSO, ACS, MEAF)")
st.write("Giao diện web trực quan để chạy thuật toán điều phối xe (VRPTW) đánh giá các bộ tham số trên Solomon dataset.")

# --- Sidebar cho cấu hình ---
st.sidebar.header("Cài đặt tham số")

dataset_dir = "data/solomon"
instances_list = []
if os.path.exists(dataset_dir):
    all_files = glob.glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True)
    # Extract just file names and capitalize (e.g., c101 -> C101)
    instances_list = sorted([os.path.basename(f).replace(".txt", "").upper() for f in all_files])

if not instances_list:
    instances_list = ["C101", "R101", "RC101"]

default_instance = ["C101"] if "C101" in instances_list else [instances_list[0]] if instances_list else []
selected_instances = st.sidebar.multiselect("Chọn Instances", options=instances_list, default=default_instance)

algorithms = [alg[0] for alg in ALGORITHMS]
selected_algs = st.sidebar.multiselect("Chọn Thuật toán", options=algorithms, default=["MEAF", "GA"])

iters = st.sidebar.number_input("Số vòng lặp (iters)", min_value=1, max_value=5000, value=500)
runs = st.sidebar.number_input("Số lần chạy (runs)", min_value=1, max_value=100, value=21)
seed = st.sidebar.number_input("Seed ngẫu nhiên", min_value=0, max_value=9999, value=42)
use_numba = st.sidebar.checkbox("Dùng Numba tăng tốc", value=True)

run_button = st.sidebar.button("Bắt đầu chạy")

if run_button:
    set_use_numba(use_numba)
    if not selected_instances:
        st.warning("Vui lòng chọn ít nhất một instance!")
    elif not selected_algs:
        st.warning("Vui lòng chọn ít nhất một thuật toán!")
    else:
        st.info("Đang chạy thuật toán... Quá trình này có thể mất vài phút.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_data = []
        
        total_steps = len(selected_instances) * len(selected_algs)
        step = 0
        
        for inst in selected_instances:
            file_path = _find_instance_file(dataset_dir, inst)
            if file_path is None or not os.path.exists(file_path):
                st.error(f"Không tìm thấy file instance: {inst}")
                continue
                
            problem = load_solomon_instance(file_path)
            
            for alg_name in selected_algs:
                # Find the corresponding function
                alg_func = next(func for name, func in ALGORITHMS if name == alg_name)
                
                status_text.text(f"Đang chạy {alg_name} trên instance {inst}...")
                
                # Execute the actual processing
                res = evaluate_algorithm(problem, alg_func, alg_name, runs, iters, seed)
                
                results_data.append({
                    "Instance": inst,
                    "Algorithms": alg_name,
                    "Best Obj": round(res["best_objective"], 4),
                    "Avg Obj": round(res["avg_objective"], 4),
                    "Std": round(res["std_objective"], 4),
                    "Distance": round(res["best_distance"], 4),
                    "Vehicles": res["vehicles_used"],
                    "Feasible": res["feasible"]
                })
                
                step += 1
                progress_bar.progress(step / total_steps)
                
        status_text.text("Hoàn thành!")
        
        # Parse output as markdown table formats
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Highlight minimum rows per instance
            st.subheader("Kết quả (Detailed Table)")
            st.dataframe(df)
            
            st.subheader("Tóm tắt (Bảng So sánh Benchmark)")
            pivot_df = df.pivot(index="Instance", columns="Algorithms", values="Best Obj")
            st.dataframe(pivot_df)

            # Option to download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Tải kết quả (CSV)", csv, "results_web.csv", "text/csv")