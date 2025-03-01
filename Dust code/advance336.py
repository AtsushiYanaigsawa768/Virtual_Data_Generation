import os
import uuid
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf, BarycentricInterpolator, KroghInterpolator, PchipInterpolator, Akima1DInterpolator
from scipy.interpolate import lagrange, BSpline, splrep, splev
from scipy.signal import savgol_filter
from scipy.special import gamma
import pywt
from pykalman import KalmanFilter
from sklearn.linear_model import RANSACRegressor
from evaluation import HAR_evaluation
from delete import delete_csv_files
from volumeChecker import check_virtual_volume
from scipy.interpolate import interp1d, CubicSpline

# ---------------------------
# 定数・グローバル変数の定義
# ---------------------------
real_directory = "/root/Virtual_Data_Generation/data/real"
train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109',
               'U0111', 'U0201', 'U0202', 'U0203', 'U0204',
               'U0206', 'U0207', 'U0208', 'U0210']
val_users = ['U0101', 'U0209']
test_users = ['U0105', 'U0110', 'U0205', 'U0209']

selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'operation',"timestamp"]

explanatory_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                       'atr02/acc_x','atr02/acc_y','atr02/acc_z']

# ---------------------------
# CSVファイルのパス収集
# ---------------------------
user_paths = {}
for root, dirs, files in os.walk(real_directory):
    for file in files:
        if file.endswith('S0100.csv'):
            # ファイル名から末尾10文字("S0100.csv")を除いた部分をユーザIDとする
            user_paths[file[:-10]] = os.path.join(root, file)
        else:
            os.remove(os.path.join(root, file))  # 不要なデータは削除

for u, d in user_paths.items():
    print(f'{u} at: {d}')

# trainユーザのデータのみ読み込み
train_data_dict = {}
for u in train_users:
    if u in user_paths:
        train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)
    else:
        print(f"User {u} not found in user_paths.")

# ---------------------------
# 各操作のセグメント長分布を計算
# ---------------------------
operation_durations = {}
for u, df in train_data_dict.items():
    df_temp = df.copy()
    # operationが変化するごとにgroup番号を付与
    df_temp['group'] = (df_temp['operation'] != df_temp['operation'].shift()).cumsum()
    for _, group_df in df_temp.groupby('group'):
        op = group_df['operation'].iloc[0]
        seg_len = len(group_df)
        operation_durations.setdefault(op, []).append(seg_len)

# IQRを使用して外れ値を除去
filtered_operation_durations = {}
for op, lengths in operation_durations.items():
    lengths_arr = np.array(lengths)
    q1 = np.percentile(lengths_arr, 25)
    q3 = np.percentile(lengths_arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 外れ値を除いたデータ
    filtered_lengths = lengths_arr[(lengths_arr >= lower_bound) & (lengths_arr <= upper_bound)]
    filtered_operation_durations[op] = filtered_lengths.tolist()
    
    # 分布の基本統計量を出力（確認用）
    print(f"Operation {op}: count={len(filtered_lengths)}, mean={filtered_lengths.mean():.2f}, "
          f"median={np.median(filtered_lengths):.2f}, std={filtered_lengths.std():.2f}, "
          f"removed={len(lengths_arr)-len(filtered_lengths)} outliers")

# 元の辞書を更新
operation_durations = filtered_operation_durations

# ---------------------------
# 補助関数の定義
# ---------------------------
def generate_unique_filename(prefix):
    """ユーザID等を接頭辞にしてユニークなファイル名を生成する"""
    return f"{prefix}_{uuid.uuid4().hex}.csv"

# --- 外れ値フィルタリング ---
def filter_moving_average(series, window=5):
    return series.rolling(window, min_periods=1, center=True).mean()

def filter_ransac(series):
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = RANSACRegressor()
    model.fit(x, y)
    y_pred = model.predict(x)
    return pd.Series(y_pred, index=series.index)

def filter_wavelet(series, wavelet='db1'):
    coeff = pywt.wavedec(series, wavelet)
    # シンプルな閾値処理
    threshold = np.median(np.abs(coeff[-1])) / 0.6745
    coeff_filtered = [coeff[0]]  # Approximation coefficients remain
    # 高周波数帯の係数に閾値処理を適用
    for c in coeff[1:]:
        coeff_filtered.append(pywt.threshold(c, threshold, mode='soft'))
    reconstructed = pywt.waverec(coeff_filtered, wavelet)
    # 元の長さに合わせて切り詰め
    reconstructed = reconstructed[:len(series)]
    return pd.Series(reconstructed, index=series.index)

def filter_kalman(series):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=series.values[0], 
                      initial_state_covariance=1,
                      observation_covariance=0.1, transition_covariance=0.1)
    state_means, _ = kf.filter(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def apply_outlier_filter(series, method='none'):
    """指定した方法で外れ値除去（必要な場合のみ）を適用"""
    if method == 'moving_average':
        return filter_moving_average(series)
    elif method == 'ransac':
        return filter_ransac(series)
    elif method == 'wavelet':
        return filter_wavelet(series)
    elif method == 'kalman':
        return filter_kalman(series)
    else:
        # 'none' の場合はフィルタリングせずそのまま返す
        return series

# --- 補間手法の定義 ---
def interpolate_linear(x, y, new_x):
    """線形補間"""
    f = interp1d(x, y, kind='linear')
    return f(new_x)

def interpolate_rbf(x, y, new_x, function='multiquadric'):
    """Radial basis function (RBF) を用いた補間"""
    rbf_func = Rbf(x, y, function=function)
    return rbf_func(new_x)

def interpolate_bspline(x, y, new_x, k=3, s=None):
    """
    B-spline補間（必要に応じてスムージング適用）。
    k: スプラインの次数 (3は三次スプライン)
    s: スムージング係数（Noneの場合はデータに基づき自動設定）
    """
    n = len(x)
    if n <= k:
        # データ点が少ない場合は低次のスプラインか線形補間を使用
        if n < 2:
            # 点が1つだけの場合は定数系列を返す
            return np.full_like(new_x, y[0])
        # 2点なら直線, 3点なら2次スプラインで補間
        tck = splrep(x, y, k=min(k, n-1), s=0)
        return splev(new_x, tck)
    # データ点が十分ある場合、平滑化スプラインを適用
    if s is None:
        # データの分散に応じてわずかなスムージングを追加 (全体の5%程度の誤差を許容)
        s = 0.05 * np.var(y) * n
    tck = splrep(x, y, k=k, s=s)
    return splev(new_x, tck)

def interpolate_akima(x, y, new_x):
    """Akima補間"""
    akima = Akima1DInterpolator(x, y)
    return akima(new_x)

def interpolate_pchip(x, y, new_x):
    """PCHIP補間"""
    pchip = PchipInterpolator(x, y)
    return pchip(new_x)

# 他の補間手法（必要に応じて定義）

def perform_interpolation(x, y, new_x, method='pchip', **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    # Filtering NaNなど必要ならここで処理する
    if method == 'interpolate':
        return interpolate_linear(x, y, new_x)
    elif method == 'rbf_inverse':
        return interpolate_rbf(x, y, new_x, function='inverse')
    elif method == 'rbf_multiquadric':
        return interpolate_rbf(x, y, new_x, function='multiquadric')
    elif method == 'rbf_gaussian':
        return interpolate_rbf(x, y, new_x, function='gaussian')
    elif method == 'rbf_linear':
        return interpolate_rbf(x, y, new_x, function='linear')
    elif method == 'rbf_cubic':
        return interpolate_rbf(x, y, new_x, function='cubic')
    elif method == 'bspline':
        return interpolate_bspline(x, y, new_x)
    elif method == 'fft_hann':
        return interpolate_fft(x, y, new_x, window='hann')
    elif method == 'fft_Welch':
        return interpolate_fft(x, y, new_x, window='Welch')
    elif method == 'fft_Blackman-Harris':
        return interpolate_fft(x, y, new_x, window='Blackman-Harris')
    elif method == 'akima':
        return interpolate_akima(x, y, new_x)
    elif method == 'pchip':
        return interpolate_pchip(x, y, new_x)
    elif method == "hermite":
        # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for Hermite interpolation
        pchip = PchipInterpolator(x, y)
        return pchip(new_x)
    elif method == 'hybrid':
        return hybrid_interpolate(x, y, new_x)
    else:
        return interpolate_linear(x, y, new_x)

def hybrid_interpolate(time, values, new_x, max_linear_gap=5):
    """
    欠損値があるシリーズに対し、短いギャップは線形補間、長いギャップはスプライン補間するハイブリッド手法。
    """
    values = np.asarray(values).copy()
    n = len(values)
    isnan = np.isnan(values)
    if not np.any(isnan):
        # 欠損が無い場合は、必要ならリサンプリングのみ
        if len(new_x) != n:
            return np.interp(new_x, time, values)
        return values
def interpolate_fft(x, y, new_x, window='hann'):
    """
    FFTベースの補間を実施する関数（概要のみ記述）
    """
    # この手法の詳細な実装は省略（元コードに存在する場合のみ使用）
    y = np.asarray(y)
    # 実装: 元の均等サンプリングをFFTし、指定の窓で周波数フィルタ後に補間
    # 簡易には線形補間で返す
    return np.interp(new_x, x, y)

# ---------------------------
# 仮想データ生成処理
# ---------------------------
def process_csv_files(output_dir, num_timestamps=None,
                      outlier_method='moving_average', 
                      interpolation_method='bspline',
                      round_values=False,
                      distribution_type='normal'):
    """
    指定した出力ディレクトリに、trainデータから生成した仮想データCSVを出力する。
    手順:
      1. train_data_dict内の各ユーザデータについて、operationが変化するごとにセグメントを抽出
      2. 各セグメントのtimestampを先頭0起点に調整
      3. 指定の外れ値除去法で加速度データをフィルタリング
      4. 指定の補間手法で各セグメントを補間し、新しいタイムスタンプ系列にリサンプリング
         - num_timestamps=Trueの場合、セグメント長を実データ分布に従いランダムにサンプリング
         - num_timestamps=Noneの場合、元のタイムスタンプを使用（補間なし）
      5. 必要に応じて値を丸め（round_values）
      6. CSVとして出力し、ディレクトリ容量が規定未満なら繰り返す
    """
    # 出力先内の既存CSVを削除
    delete_csv_files(output_dir)
    rootdir = '/root/Virtual_Data_Generation'
    virtpath = '/data/virtual'
    # 出力先容量が500MB未満になるまで繰り返し生成
    while check_virtual_volume(rootdir, virtpath, limit_mb=500) is True:
        for user, df in train_data_dict.items():
            # データフレームをコピーし必要列のみ抽出
            df_proc = df[selected_columns].copy()
            # operationの変化毎にグループIDを作成
            df_proc['group'] = (df_proc['operation'] != df_proc['operation'].shift()).cumsum()
            processed_groups = []
            # 各グループ（連続した同一operation区間）ごとに処理
            for group_id, group_df in df_proc.groupby('group'):
                group_df = group_df.copy()
                # タイムスタンプを先頭0に調整
                group_df['timestamp'] = group_df['timestamp'] - group_df['timestamp'].iloc[0]
                # 新しい timestamp 配列を生成
                if num_timestamps:
                    op_val = group_df['operation'].iloc[0]
                    if op_val in operation_durations:
                        # 実データの分布からセグメント長をサンプリング
                        durations = np.array(operation_durations[op_val])
                        mean_val = durations.mean()
                        std_val = durations.std()
                        # 分布タイプに基づいてサンプリング
                        current_num = 0
                        current_num = 0
                        
                        while current_num < 2:  # 最低2ポイントは必要
                            if distribution_type == 'normal':
                                current_num = int(np.random.normal(mean_val, std_val))
                            elif distribution_type == 'gamma':
                                shape = (mean_val / std_val) ** 2 if std_val > 0 else 1.0
                                scale = (std_val ** 2) / mean_val if mean_val > 0 else 1.0
                                current_num = int(np.random.gamma(shape, scale))
                            elif distribution_type == 'exponential':
                                lambda_param = 1.0 / mean_val if mean_val > 0 else 1.0
                                current_num = int(np.random.exponential(1.0 / lambda_param))
                            elif distribution_type == 'lognormal':
                                # Convert normal mean/std to lognormal parameters
                                mu = np.log(mean_val**2 / np.sqrt(std_val**2 + mean_val**2)) if mean_val > 0 else 0
                                sigma = np.sqrt(np.log(1 + (std_val**2 / mean_val**2))) if mean_val > 0 else 1
                            elif distribution_type == 'weibull':
                                # Approximation for Weibull parameters
                                k = (std_val / mean_val)**-1.086 if mean_val > 0 else 1.0
                                lambda_weibull = mean_val / gamma(1 + 1/k) if k > 0 else mean_val
                                current_num = int(lambda_weibull * np.random.weibull(k))
                                current_num = int(lambda_weibull * np.random.weibull(k))
                            elif distribution_type == 'poisson':
                                current_num = np.random.poisson(mean_val)
                            else:  # Default to normal
                                current_num = int(np.random.normal(mean_val, std_val))
                    else:
                        current_num = max(len(group_df), 2)
                        
                    # Ensure we have at least 2 points
                    current_num = max(current_num, 2)
                    
                    new_timestamps = np.linspace(group_df['timestamp'].min(),
                                                 group_df['timestamp'].max(),
                                                 current_num)
                else:
                    new_timestamps = group_df['timestamp'].values
                # 新しいグループ用のDataFrameを作成
                new_group_df = pd.DataFrame({'timestamp': new_timestamps})
                # 各説明変数に対して外れ値フィルタ適用と補間
                for col in explanatory_columns:
                    filtered_series = apply_outlier_filter(group_df[col], method=outlier_method)
                    interp_values = perform_interpolation(group_df['timestamp'].values,
                                                         filtered_series.values,
                                                         new_timestamps,
                                                         method=interpolation_method)
                    new_group_df[col] = interp_values
                # operation列を付与
                new_group_df['operation'] = group_df['operation'].iloc[0]
                processed_groups.append(new_group_df)
            # 全グループ処理後、連結
            final_df = pd.concat(processed_groups, ignore_index=True)
            # 値を必要に応じて丸め
            if round_values:
                final_df = final_df.round(5)
            final_df = final_df.drop(columns=['timestamp'])  # timestamp列は不要なので削除
            # ユニークなファイル名で保存
            output_file = os.path.join(output_dir, generate_unique_filename(user))
            final_df.to_csv(output_file, index=False)
            # 生成データ容量チェック
            if check_virtual_volume(rootdir, virtpath, limit_mb=500) is False:
                return None

# ---------------------------
# 実行例
# ---------------------------
def run_virtual_data_generation():
    distribution_types = [ 'weibull', 'poisson']
    results = {}
    
    for dist_type in distribution_types:
        f1list = []
        print(f"Testing distribution type: {dist_type}")
        
        # Add distribution type as a parameter to process_csv_files function
        def process_with_distribution(dist):
            # Modify the process_csv_files to use the specific distribution
            output_dir = f'/root/Virtual_Data_Generation/data/virtual'
            process_csv_files(output_dir, num_timestamps=True,
                              outlier_method='none',
                              interpolation_method='rbf_inverse',
                              round_values=True,
                              distribution_type=dist)
            return output_dir
        for i in range(1):
            output_dir = process_with_distribution(dist_type)
            f1 = HAR_evaluation(f"distribution_{dist_type}_{1}", output_dir)
            print(f"Generated virtual data with {dist_type} distribution: F1={f1}")
            f1list.append(f1)
            
        avg_f1 = sum(f1list) / len(f1list)
        results[dist_type] = {"scores": f1list, "average": avg_f1}
        print(f"{dist_type} distribution results: {f1list}, average: {avg_f1}")
    
    # Print overall results sorted by average F1 score
    print("\n=== RESULTS SUMMARY ===")
    for dist, data in sorted(results.items(), key=lambda x: x[1]["average"], reverse=True):
        print(f"{dist}: {data['average']:.4f} - {data['scores']}")
    return results

if __name__ == '__main__':
    run_virtual_data_generation()
