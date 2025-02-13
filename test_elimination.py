import os
import pandas as pd
import numpy as np
import random
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from volumeChecker import check_virtual_volume
from scipy.optimize import curve_fit
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
                if pos == 0:
                    lower_idx, upper_idx = 0, 1
                elif pos >= len(ts_array):
                    lower_idx, upper_idx = len(ts_array) - 2, len(ts_array) - 1
                else:
                    lower_idx, upper_idx = pos - 1, pos
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
        rbfs[col] = Rbf(ts_array, df_grouped[col].values, function='multiquadric', epsilon=1)
    
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
            cubic_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
            row[col] = float(cubic_interp(rt))
        results.append(row)
    return results

def curve_fit_implementation(random_times, df_grouped):
    # Define a simple linear function for curve fitting
    def func(x, a, b, c, d, e, f, g):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

    results = []
    ts_array = df_grouped['timestamp'].values

    # Precompute curve fit parameters for all columns except 'timestamp'
    popt_dict = {}
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        try:
            popt, _ = curve_fit(func, ts_array, df_grouped[col].values)
        except Exception as e:
            popt = np.polyfit(ts_array, df_grouped[col].values, 1)
        popt_dict[col] = popt

    for rt in random_times:
        row = {"timestamp": rt}
        for col, popt in popt_dict.items():
            row[col] = func(rt, *popt)
        results.append(row)
    return results

def fft_interpolate(fs, amp_array, time_array, k=0):
    N = len(amp_array)
    freq_array = np.fft.rfftfreq(N, d=1/fs)
    fft_amp_array = 2.0 / N * np.fft.rfft(amp_array)
    fft_amp_array[0] /= 2.0
    fft_amp_array = 1j**k * fft_amp_array
    an = np.real(fft_amp_array)
    bn = -np.imag(fft_amp_array)
    factor = (2 * np.pi * freq_array)**k
    # Vectorized computation across all frequencies and time points
    cos_vals = np.cos(2 * np.pi * np.outer(freq_array, time_array))
    sin_vals = np.sin(2 * np.pi * np.outer(freq_array, time_array))
    amp_array_interp = (factor[:, None] * (an[:, None] * cos_vals + bn[:, None] * sin_vals)).sum(axis=0)
    return amp_array_interp

def fft_implementation(random_times, df_grouped):
    ts_array = df_grouped['timestamp'].values
    if len(ts_array) > 1:
        dt = np.mean(np.diff(ts_array))
        fs = 1.0 / dt if dt != 0 else 1.0
    else:
        fs = 1.0

    rt_array = np.array(random_times)
    results = [{"timestamp": rt} for rt in random_times]

    # Pre-compute interpolated values for each column at once
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        amp_array = df_grouped[col].values
        interpolated_vals = fft_interpolate(fs, amp_array, rt_array, k=0)
        for idx, val in enumerate(interpolated_vals):
            results[idx][col] = float(val)
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

    # repeat_int個の要素 (t_min と t_max を含む) の等間隔のタイムスタンプを生成し、各値にN(0,1)のノイズを加える
    base_times = np.linspace(t_min, t_max, repeat_int)
    noise = np.random.normal(0, 1, repeat_int)
    random_times = (base_times + noise).tolist()
    # 重複を避けるために、ソートしてリストで保持
    random_times = sorted(random_times)
    if method == 'linear':
        results = interpolate_results(random_times, df_grouped)
    elif method == 'rbf':
        results = rbf_adapt(random_times, df_grouped)
    elif method == 'cubic':
        results = cubic_implementation(random_times, df_grouped)
    elif method == 'curve_fit':
        results = curve_fit_implementation(random_times, df_grouped)
    elif method == 'fft':
        results = fft_implementation(random_times, df_grouped)
    else:
        raise ValueError("Invalid interpolation method")
    for result in results:
        result["operation"] = float(operation_int)
    df_results = pd.DataFrame(results)
    # 降順ソート(必要に応じて)
    df_results = df_results.sort_values('timestamp', ascending=True)
    
    return df_results


def main_loop(operation_int,repeat_int,volume_limit=500,method='linear'):
    output_csv = rootdir +  r'/data/test3' + f'/interpolated_{operation_int}.csv'
    # 既存の出力ファイルがあれば削除しておく
    if os.path.exists(output_csv):
        os.remove(output_csv)
    volume_checker = True
    while volume_checker:
        df_results = read_and_interpolate(operation_int, repeat_int,method=method)
        # timestamp列を除外して保存
        df_to_save = df_results.drop(columns=["timestamp"])
        
        # 既存のCSVに追記（ファイルがなければヘッダー付きで作成）
        df_to_save.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)       
        # volumeCheckerを利用して現在のvolumeを確認
        volume_checker = check_virtual_volume(rootdir,r'/data/test3',volume_limit)

if __name__ == "__main__":
    i = 0
    for operation_int in [100,200,300,400,500,600,700,800,900,1000,8100]:
        i += 1
        main_loop(operation_int=operation_int,repeat_int=300,volume_limit=i * 10,method='fft')