import os
import pandas as pd
import numpy as np
import random
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from volumeChecker import check_virtual_volume
rootdir = r'/root/Virtual_Data_Generation'
virtpath = r'/data/test'
def interpolate_results(random_times, df_grouped):
    results = []
    ts_array = df_grouped['timestamp'].values
    for rt in random_times:
        row = {"timestamp": rt}
        for col in df_grouped.columns:
            if col == "timestamp":
                continue
            if rt in ts_array:
                value = df_grouped.loc[df_grouped['timestamp'] == rt, col].mean()
            else:
                pos = np.searchsorted(ts_array, rt)
                lower_idx = pos - 1
                upper_idx = pos
                t0 = ts_array[lower_idx]
                t1 = ts_array[upper_idx]
                y0 = df_grouped.loc[df_grouped['timestamp'] == t0, col].mean()
                y1 = df_grouped.loc[df_grouped['timestamp'] == t1, col].mean()
                value = y0 + (y1 - y0) * (rt - t0) / (t1 - t0)
            row[col] = value
        results.append(row)
    return results

def rbf_adapt(random_times, df_grouped):
    results = []
    ts_array = df_grouped['timestamp'].values
    # Precompute Rbf interpolators for all columns except 'timestamp'
    rbfs = {}
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        rbfs[col] = Rbf(ts_array, df_grouped[col].values, function='thin_plate', epsilon=1)
    
    for rt in random_times:
        row = {"timestamp": rt}
        for col, rbf in rbfs.items():
            # Compute the interpolated value at rt
            row[col] = rbf(rt)
        results.append(row)
    return results

def cubic_implementation(random_times, df_grouped):
    results = []
    ts_array = df_grouped['timestamp'].values
    for rt in random_times:
        row = {"timestamp": rt}
        for col in df_grouped.columns:
            if col == "timestamp":
                continue
            x = ts_array
            y = df_grouped[col].values
            # Create a cubic interpolator with extrapolation enabled
            cubic_interp = interp1d(x, y, kind='cubic')
            row[col] = float(cubic_interp(rt))
        results.append(row)
    return results
def read_and_interpolate(operation_int, repeat_int,method='linear'):
    # CSVファイルのパス
    csv_path = f"/root/Virtual_Data_Generation/data/converted4/{operation_int}.csv"
    
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)

    # 同じtimestampが複数ある場合は平均を取る
    df_grouped = df.groupby('timestamp', as_index=False).mean()
    
    # timestampで昇順にソート
    df_grouped = df_grouped.sort_values('timestamp')
    
    # timestampの最小値と最大値を取得
    t_min = df_grouped['timestamp'].min()
    t_max = df_grouped['timestamp'].max()

    # repeat_int個のランダムなタイムスタンプを生成（t_min, t_max は必ず含む）
    random_times = [t_min, t_max]
    for _ in range(repeat_int -2):
        random_times.append(random.uniform(t_min, t_max))
    
    # 重複を避けるために、ソートしてリストで保持
    random_times = sorted(random_times)
    if method == 'linear':
        results = interpolate_results(random_times, df_grouped)
    elif method == 'rbf':
        results = rbf_adapt(random_times, df_grouped)
    elif method == 'cubic':
        results = cubic_implementation(random_times, df_grouped)
    else:
        raise ValueError("Invalid interpolation method")
    for result in results:
        result["operation"] = float(operation_int)
    df_results = pd.DataFrame(results)
    # 降順ソート(必要に応じて)
    df_results = df_results.sort_values('timestamp', ascending=True)
    
    return df_results


def main_loop(operation_int,repeat_int,volume_limit=500,method='linear'):
    output_csv = rootdir +  r'/data/test' + f'/interpolated_{operation_int}.csv'
    # 既存の出力ファイルがあれば削除しておく
    if os.path.exists(output_csv):
        os.remove(output_csv)
    volume_checker = True
    import time
    while volume_checker:
        df_results = read_and_interpolate(operation_int, repeat_int, method=method)
        # timestamp列を除外して保存
        df_to_save = df_results.drop(columns=["timestamp"])
        
        # 既存のCSVに追記（ファイルがなければヘッダー付きで作成）
        df_to_save.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)       
        # volumeCheckerを利用して現在のvolumeを確認
        volume_checker = check_virtual_volume(rootdir, r'/data/test', volume_limit)
        # Sleep to prevent high CPU usage and excessive resource consumption
        time.sleep(1)

if __name__ == "__main__":
    i = 0
    for operation_int in [100,200,300,400,500,600,700,800,900,1000,8100]:
        i += 1
        main_loop(operation_int=operation_int,repeat_int=300,volume_limit=i * 10,method='cubic')