import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 設定値
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'timestamp','operation']
acc_cols = selected_columns[:6]
rootdir = r'/root/Virtual_Data_Generation'
real_path = os.path.join(rootdir, 'data', 'real')

def process_file(csv_file):
    try:
        df = pd.read_csv(csv_file, usecols=selected_columns)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return pd.DataFrame()

    # 連続するoperationのグループを作成（operation変化毎にgroup idを付与）
    df['group'] = (df['operation'] != df['operation'].shift()).cumsum()

    # 各グループ内でt=0となるtimestampを取得し、elapsed timeを計算
    df['elapsed'] = df.groupby('group')['timestamp'].transform(lambda x: x - x.iloc[0])
    return df[['operation', 'elapsed'] + acc_cols]

# realディレクトリ内の全CSVファイルを取得
csv_files = glob.glob(os.path.join(real_path, '*.csv'))
if not csv_files:
    print("CSVファイルが見つかりません")
    exit()

# 並列処理で各CSVファイルを処理
dfs = []
with ThreadPoolExecutor() as executor:
    results = executor.map(process_file, csv_files)
    for res in results:
        if not res.empty:
            dfs.append(res)
if len(dfs) == 0:
    print("有効なデータがありません")
    exit()

# 全ファイルのデータを結合
all_data = pd.concat(dfs, ignore_index=True)

# elapsed time毎の平均・分散を operation 別に算出
grouped = all_data.groupby(['operation', 'elapsed'])
stats = grouped.agg([np.mean, np.var])
stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
stats = stats.reset_index()

# 各operation全体の統計を計算（elapsed timeは 'overall' として）
overall = all_data.groupby('operation')[acc_cols].agg([np.mean, np.var])
overall.columns = ['_'.join(col).strip() for col in overall.columns.values]
overall = overall.reset_index()
overall.insert(1, 'elapsed', 'overall')

# 結果を結合してCSV出力
final_stats = pd.concat([stats, overall], ignore_index=True)
csv_output = os.path.join(rootdir, 'duration_inquarity.csv')
final_stats.to_csv(csv_output, index=False)

print(f"集計結果を {csv_output} に保存しました。")