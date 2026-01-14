import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import zipfile

# ================= CONFIGURATION =================
DATA_DIR = '/kyb/agpd/wwang/data/sst_valid_base'
OUT_DIR = '/kyb/agpd/wwang/sst-pomdp/outputs/'
OUTPUT_CSV = os.path.join(OUT_DIR, 'se_summary_detailed.csv')
OUTPUT_PLOT_DIST = os.path.join(OUT_DIR, 'se_phase_dist.png') 
OUTPUT_PLOT_RT_SSD = os.path.join(OUT_DIR, 'se_rt_by_ssd.png') 

STOP_SIGNAL_DURATION = 300  # ms
MODEL_HORIZON = 1000        # ms (1s)
# =================================================

def analyze_sst_errors_final():
    zip_files = glob.glob(os.path.join(DATA_DIR, '*.zip'))
    print(f"Processing {len(zip_files)} subjects...")

    results = []
    all_se_data = [] 
    global_counts = {'Premature': 0, 'Signal': 0, 'Fixation': 0, 'Late': 0, 'Total_Errors': 0}

    for zip_path in zip_files:
        try:
            base_name = os.path.basename(zip_path)
            subject_id = base_name.split('_')[1] if 'NDAR_' in base_name else base_name.split('.')[0]
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
                if not csv_files: continue
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)

            se_df = df[(df['sst_expcon'] == 'VariableStopTrial') & (df['sst_inhibitacc'] == 0)].copy()
            if se_df.empty: continue

            for s, r in zip(se_df['sst_ssd_dur'], se_df['sst_primaryrt']):
                all_se_data.append({'SSD': s, 'RT': r})
                # 阶段统计
                if r < s: global_counts['Premature'] += 1
                elif r <= s + STOP_SIGNAL_DURATION: global_counts['Signal'] += 1
                elif r <= MODEL_HORIZON: global_counts['Fixation'] += 1
                else: global_counts['Late'] += 1
                global_counts['Total_Errors'] += 1

        except Exception: continue

    # --- 1. 绘制阶段分布直方图 (Phase Distribution) ---
    plt.figure(figsize=(8, 6))
    labels = ['Premature', 'Signal', 'Fixation', 'Late']
    counts = [global_counts[k] for k in labels]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    bars = plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    plt.title('Global Distribution of Stop Error Phases', fontsize=14)
    plt.ylabel('Trial Count')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{(height/global_counts["Total_Errors"])*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.savefig(OUTPUT_PLOT_DIST)
    print(f"Saved: {OUTPUT_PLOT_DIST}")

    # --- 2. 绘制垂直 KDE 分布图 (RT by SSD) ---
    df_plot = pd.DataFrame(all_se_data)
    # 过滤出样本量足够的 SSD 以免图像太乱
    common_ssds = df_plot['SSD'].value_counts()[df_plot['SSD'].value_counts() > 30].index.sort_values()
    df_filtered = df_plot[df_plot['SSD'].isin(common_ssds)]

    plt.figure(figsize=(15, 8))
    
    # 使用小提琴图展示垂直 KDE (inner=None 去掉内部的中位数线，更纯粹展示分布)
    ax = sns.violinplot(data=df_filtered, x='SSD', y='RT', inner=None, 
                        color='lightgrey', cut=0, density_norm='width')

    # 绘制边界线
    for i, s_val in enumerate(common_ssds):
        # 蓝色线：SSD Onset (停止信号开始)
        plt.hlines(y=s_val, xmin=i-0.4, xmax=i+0.4, color='blue', linewidth=2.5, 
                   label='SSD (Signal On)' if i==0 else "")
        
        # 橙色线：SSD + 300ms (停止信号结束)
        if s_val + STOP_SIGNAL_DURATION < MODEL_HORIZON:
            plt.hlines(y=s_val + STOP_SIGNAL_DURATION, xmin=i-0.4, xmax=i+0.4, 
                       color='orange', linewidth=2.5, label='SSD + 300ms (Signal Off)' if i==0 else "")

    plt.axhline(y=MODEL_HORIZON, color='red', linestyle='--', linewidth=1.5, label='1s Limit')
    
    plt.title('Stop Error RT Distribution by SSD', fontsize=15)
    plt.ylabel('Reaction Time (ms)', fontsize=12)
    plt.xlabel('Stop Signal Delay (ms)', fontsize=12)
    plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_RT_SSD)
    print(f"Saved: {OUTPUT_PLOT_RT_SSD}")

if __name__ == "__main__":
    analyze_sst_errors_final()