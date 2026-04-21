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
max_customers = st.sidebar.number_input("Giới hạn số khách (demo)", min_value=0, max_value=100, value=0)
init_heuristic = st.sidebar.checkbox("Khởi tạo bằng heuristic", value=True)
show_runs = st.sidebar.checkbox("Hiển thị chi tiết runs/seed", value=False)
show_convergence = st.sidebar.checkbox("Hiển thị biểu đồ hội tụ", value=False)

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
        history_map = {}
        
        total_steps = len(selected_instances) * len(selected_algs)
        step = 0
        
        for inst in selected_instances:
            file_path = _find_instance_file(dataset_dir, inst)
            if file_path is None or not os.path.exists(file_path):
                st.error(f"Không tìm thấy file instance: {inst}")
                continue
            
            problem = load_solomon_instance(file_path, max_customers=max_customers if max_customers > 0 else None)
            
            for alg_name in selected_algs:
                # Find the corresponding function
                alg_func = next(func for name, func in ALGORITHMS if name == alg_name)
                
                status_text.text(f"Đang chạy {alg_name} trên instance {inst}...")
                
                # Execute the actual processing
                res = evaluate_algorithm(
                    problem,
                    alg_func,
                    alg_name,
                    runs,
                    iters,
                    seed,
                    return_runs=show_runs,
                    return_history=show_convergence,
                    init_heuristic=init_heuristic,
                )
                
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

                if show_runs and res.get("runs"):
                    runs_df = pd.DataFrame(res["runs"])
                    st.caption(f"Runs/seed cho {alg_name} - {inst}")
                    st.dataframe(runs_df)

                if show_convergence and res.get("history"):
                    hist = res["history"]
                    hist_df = pd.DataFrame({"iteration": list(range(1, len(hist) + 1)), alg_name: hist})
                    st.caption(f"Hội tụ {alg_name} - {inst}")
                    st.line_chart(hist_df.set_index("iteration"))
                    history_map[(inst, alg_name)] = hist

        if show_convergence and results_data:
            st.subheader("Biểu đồ hội tụ gộp (giống Fig.4)")
            for inst in selected_instances:
                combined = None
                for alg_name in selected_algs:
                    hist = history_map.get((inst, alg_name))
                    if hist:
                        series = pd.Series(hist, name=alg_name)
                        if combined is None:
                            combined = pd.DataFrame({"iteration": list(range(1, len(hist) + 1))})
                        combined[alg_name] = series.values

                if combined is not None:
                    st.caption(f"{inst}")
                    st.line_chart(combined.set_index("iteration"))
                
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

            st.subheader("Gợi ý thuyết trình (tự động)")
            for inst in df["Instance"].unique():
                inst_df = df[df["Instance"] == inst]
                best_row = inst_df.loc[inst_df["Avg Obj"].idxmin()]
                worst_row = inst_df.loc[inst_df["Avg Obj"].idxmax()]
                feasible_ratio = inst_df["Feasible"].mean()
                st.markdown(
                    f"- **{inst}**: tốt nhất theo Avg là **{best_row['Algorithms']}** (Avg={best_row['Avg Obj']:.2f}, Std={best_row['Std']:.2f}); "
                    f"kém nhất là **{worst_row['Algorithms']}** (Avg={worst_row['Avg Obj']:.2f}). "
                    f"Tỷ lệ feasible: {feasible_ratio:.0%}."
                )

            # Option to download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Tải kết quả (CSV)", csv, "results_web.csv", "text/csv")