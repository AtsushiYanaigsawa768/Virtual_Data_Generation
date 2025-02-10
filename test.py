import os
import re
import glob
import pandas as pd

# ディレクトリのパス
data_dir = '/root/Virtual_Data_Generation/data/virtual'

# CSVファイルの一覧を取得
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# operation毎にファイルをグループ化するための辞書
operation_dict = {}

# ファイル名のパターン: "generated_<operation>_..."
pattern = re.compile(r'^generated_([^_]+)_')

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    m = pattern.search(file_name)
    if m:
        operation = m.group(1)
        operation_dict.setdefault(operation, []).append(file_path)

# 各operation毎にCSVファイルを結合して保存し、元のファイルを削除
for operation, files in operation_dict.items():
    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    # 全てのデータフレームを縦方向に連結 (共通のラベル＝カラム)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 結合結果を書き込むファイル名（例: combined_<operation>.csv）
    output_file = os.path.join(data_dir, f'combined_{operation}.csv')
    combined_df.to_csv(output_file, index=False)
    print(f'Created: {output_file}')
    
    # 元のCSVファイルを削除
    for file_path in files:
        os.remove(file_path)
        print(f'Deleted: {file_path}')