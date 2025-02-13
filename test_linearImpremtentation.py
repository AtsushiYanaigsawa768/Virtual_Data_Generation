import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, Rbf
from scipy.optimize import curve_fit
from evaluation import HAR_evaluation
from delete import delete_csv_files
from volumeChecker import check_virtual_volume
realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/root/Virtual_Data_Generation'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
# --- 補間関数群 ---
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
    rbfs = {}
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        rbfs[col] = Rbf(ts_array, df_grouped[col].values, function='multiquadric', epsilon=1)
    
    for rt in random_times:
        row = {"timestamp": rt}
        for col, rbf in rbfs.items():
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
            cubic_interp = interp1d(ts_array, df_grouped[col].values, kind='cubic', fill_value="extrapolate")
            row[col] = float(cubic_interp(rt))
        results.append(row)
    return results

def curve_fit_implementation(random_times, df_grouped):
    def func(x, a, b, c, d, e, f, g):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

    results = []
    ts_array = df_grouped['timestamp'].values
    popt_dict = {}
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        try:
            popt, _ = curve_fit(func, ts_array, df_grouped[col].values)
        except Exception:
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

    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        amp_array = df_grouped[col].values
        interpolated_vals = fft_interpolate(fs, amp_array, rt_array, k=0)
        for idx, val in enumerate(interpolated_vals):
            results[idx][col] = float(val)
    return results

# --- 補間処理のメイン関数 ---
def process_interpolation(input_dir, output_dir, selected_columns, explanatory_columns,iteration,method):
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    for csv_file in csv_files:
        # CSVの読み込みと必要な列の抽出
        df = pd.read_csv(csv_file)
        df = df[selected_columns]
    
        # operationが変わるタイミングで区間（グループ）を設定
        df['group'] = (df['operation'] != df['operation'].shift()).cumsum()
    
        all_interpolated = []  # 各区間の補間結果を保持するリスト
    
        for group_id, group_df in df.groupby('group'):
            group_df = group_df.reset_index(drop=True)
        
            # 1. タイムスタンプのリセット：最初の値を0にする
            start_time = group_df.loc[0, 'timestamp']
            group_df['timestamp'] = group_df['timestamp'] - start_time
        
            num_points = len(group_df)
        
            # 4. [0, 最終timestamp] の範囲から num_points 個の時刻をランダムに抽出（昇順にソート）
            if num_points > 1:
                new_times = np.sort(np.random.uniform(0, group_df['timestamp'].iloc[-1], num_points))
            else:
                new_times = group_df['timestamp'].values
        
            # 補間対象は timestamp と各説明変数
            interp_df = group_df[['timestamp'] + explanatory_columns]
        
            if method == 'linear':
                interp_result  = interpolate_results(new_times, interp_df)
            elif method == 'rbf':
                interp_result  = rbf_adapt(new_times, interp_df)
            elif method == 'cubic':
                interp_result  = cubic_implementation(new_times, interp_df)
            elif method == 'curve_fit':
                interp_result  = curve_fit_implementation(new_times, interp_df)
            elif method == 'fft':
                interp_result  = fft_implementation(new_times, interp_df)
            else:
                raise ValueError("Invalid interpolation method")
            # 補間結果に元の operation の値を追加
            for row in interp_result:
                row['operation'] = group_df.loc[0, 'operation']
        
            all_interpolated.extend(interp_result)
    
        # 補間結果をDataFrameに変換、出力（timestamp列は除去）
        result_df = pd.DataFrame(all_interpolated).drop(columns=['timestamp'])
        output_file = os.path.join(output_dir, f"{os.path.basename(csv_file)}_{iteration}.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}")

# --- ある条件が整うまで生成を続けるメインループ ---
def generate_until_condition(input_dir, output_dir, selected_columns, explanatory_columns, i):
    iteration = 0
    delete_csv_files(virt_directory)
    volume_checker = True
    while volume_checker:
        print(f"Iteration: {iteration}")
        process_interpolation(input_dir, output_dir, selected_columns, explanatory_columns,iteration,i)
        print("Iteration completed. Checking condition...\n")
        iteration += 1
        volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500)
    F1score = HAR_evaluation(iteration)
    print("Condition met. Generation stopped.")
    return F1score

# --- 呼び出し例 ---
if __name__ == "__main__":
    input_dir = '/root/Virtual_Data_Generation/data/real'
    output_dir = '/root/Virtual_Data_Generation/data/virtual'
    selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                        'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                        'timestamp','operation']
    explanatory_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                           'atr02/acc_x','atr02/acc_y','atr02/acc_z']
    F1score_list = []
    for i in ['linear','cubic','curve_fit','fft',"rbf"]:
        # 条件関数 condition_met を満たすまで処理を繰り返す
        F1score = generate_until_condition(input_dir, output_dir, selected_columns, explanatory_columns,i)
        F1score_list.append(F1score)
    print(F1score_list)