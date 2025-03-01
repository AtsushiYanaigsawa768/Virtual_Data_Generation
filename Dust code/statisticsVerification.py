import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# 設定値
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'timestamp','operation']
acc_cols = selected_columns[:6]
rootdir = r'/root/Virtual_Data_Generation'  # プロジェクトルート
real_path = os.path.join(rootdir, 'data', 'real')
stats_dir = os.path.join(rootdir, 'Statistics')
os.makedirs(stats_dir, exist_ok=True)  # Statisticsフォルダ作成

# CSVファイルのパス一覧（ファイル名が"S0100.csv"で終わるもの）
file_paths = []
for root, dirs, files in os.walk(real_path):
    for file in files:
        if file.endswith('S0100.csv'):
            file_paths.append(os.path.join(root, file))
        else:
            os.remove(os.path.join(root, file))  # 不要なファイルは削除

def process_file(filepath):
    """1ファイル分の統計情報を算出"""
    local_stats = defaultdict(lambda: {"frequency": 0,
                                       "durations": [],
                                       "time_stats": defaultdict(list)})
    try:
        df = pd.read_csv(filepath, usecols=selected_columns)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return local_stats
    # timestampは数値に変換しソート
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    current_label = None
    segment_start = 0
    for i, row in df.iterrows():
        op = row['operation']
        if current_label is None:
            current_label = op
            segment_start = i
        elif op != current_label:
            segment = df.iloc[segment_start:i].reset_index(drop=True)
            local_stats[current_label]["frequency"] += 1
            # 変更: timestampの変化量ではなく、固定値1（列数ではなく1に固定）
            duration = 1  
            local_stats[current_label]["durations"].append(duration)
            for offset, seg_row in segment.iterrows():
                acc_values = seg_row[acc_cols].astype(float).values
                local_stats[current_label]["time_stats"][offset].append(acc_values)
            current_label = op
            segment_start = i
    if current_label is not None and segment_start < len(df):
        segment = df.iloc[segment_start:].reset_index(drop=True)
        local_stats[current_label]["frequency"] += 1
        # 変更: timestampの変化量ではなく、固定値1
        duration = 1  
        local_stats[current_label]["durations"].append(duration)
        for offset, seg_row in segment.iterrows():
            acc_values = seg_row[acc_cols].astype(float).values
            local_stats[current_label]["time_stats"][offset].append(acc_values)
    return local_stats


def merge_stats(global_stats, local_stats):
    """個々のファイル結果を集約"""
    for label, data in local_stats.items():
        global_stats[label]["frequency"] += data["frequency"]
        global_stats[label]["durations"].extend(data["durations"])
        for offset, values in data["time_stats"].items():
            global_stats[label]["time_stats"][offset].extend(values)

# 並列処理で全ファイルを処理
global_stats = defaultdict(lambda: {"frequency": 0,
                                    "durations": [],
                                    "time_stats": defaultdict(list)})
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, fp) for fp in file_paths]
    for future in as_completed(futures):
        local = future.result()
        merge_stats(global_stats, local)

# CSV出力用リスト（summary行とタイムオフセットの詳細行を含む）
csv_rows = []

# ラベル出現頻度・平均継続時間・継続時間の分散の算出（summary）
for label, data in global_stats.items():
    freq = data["frequency"]
    avg_dur = np.mean(data["durations"]) if data["durations"] else 0
    dur_var = np.var(data["durations"]) if data["durations"] else 0
    print("Summary - Label: {}, Frequency: {}, Average Duration: {}, Duration Variance: {}".format(
        label, freq, avg_dur, dur_var))

# # タイムオフセットごとの平均・分散情報の算出（詳細）
# for label, data in global_stats.items():
#     time_stats = data["time_stats"]
#     if not time_stats:
#         continue
#     max_offset = max(time_stats.keys())
#     for offset in range(max_offset + 1):
#         if offset in time_stats and len(time_stats[offset]) > 0:
#             arr = np.array(time_stats[offset])
#             means = np.mean(arr, axis=0)
#             vars_ = np.var(arr, axis=0)
#         else:
#             means = [np.nan] * len(acc_cols)
#             vars_  = [np.nan] * len(acc_cols)
#         row = {
#             "row_type": "offset",
#             "label": label,
#             "offset": offset
#         }
#         for i, col in enumerate(acc_cols):
#             row[f"{col}_mean"] = means[i]
#             row[f"{col}_var"] = vars_[i]
#         csv_rows.append(row)

# # CSVへ書き出し（全結果を1ファイルに保存）
# csv_output_path = os.path.join(stats_dir, "Statistics.csv")
# df_out = pd.DataFrame(csv_rows)
# df_out.to_csv(csv_output_path, index=False)
# print(f"統計情報は {csv_output_path} に保存されました。")

# # プロットの生成と保存（各ラベルごとにタイムオフセットの平均・分散推移）
# for label, data in global_stats.items():
#     time_stats = data["time_stats"]
#     if not time_stats:
#         continue
#     max_offset = max(time_stats.keys())
#     offsets = list(range(max_offset + 1))
#     means_all = {i: [] for i in range(len(acc_cols))}
#     vars_all  = {i: [] for i in range(len(acc_cols))}
#     for offset in offsets:
#         if offset in time_stats and len(time_stats[offset]) > 0:
#             arr = np.array(time_stats[offset])
#             means = np.mean(arr, axis=0)
#             vars_ = np.var(arr, axis=0)
#         else:
#             means = [np.nan] * len(acc_cols)
#             vars_  = [np.nan] * len(acc_cols)
#         for i in range(len(acc_cols)):
#             means_all[i].append(means[i])
#             vars_all[i].append(vars_[i])
            
#     # 平均の推移をプロット
#     plt.figure(figsize=(12, 6))
#     for i, col in enumerate(acc_cols):
#         plt.plot(offsets, means_all[i], label=f'{col} mean')
#     plt.xlabel("Time Offset")
#     plt.ylabel("Average")
#     plt.title(f"Label {label} - Mean Transition")
#     plt.legend()
#     plt.grid(True)
#     mean_fig_path = os.path.join(stats_dir, f"{label}_mean.png")
#     plt.savefig(mean_fig_path)
#     plt.close()
    
#     # 分散の推移をプロット
#     plt.figure(figsize=(12, 6))
#     for i, col in enumerate(acc_cols):
#         plt.plot(offsets, vars_all[i], label=f'{col} var')
#     plt.xlabel("Time Offset")
#     plt.ylabel("Variance")
#     plt.title(f"Label {label} - Variance Transition")
#     plt.legend()
#     plt.grid(True)
#     var_fig_path = os.path.join(stats_dir, f"{label}_variance.png")
#     plt.savefig(var_fig_path)
#     plt.close()
    
#     print(f"{label} のプロット画像が {mean_fig_path} と {var_fig_path} として保存されました。")