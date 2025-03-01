import os
import uuid
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf, PchipInterpolator, Akima1DInterpolator
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
import pywt
from pykalman import KalmanFilter
from sklearn.linear_model import RANSACRegressor
from evaluation import HAR_evaluation
from delete import delete_csv_files
from volumeChecker import check_virtual_volume
from scipy.interpolate import interp1d, CubicSpline
import glob
# ---------------------------
# 定数・グローバル変数の定義
# ---------------------------
real_directory = "/root/Virtual_Data_Generation/data/real"
# ※ selected_columns に 'action' を追加（evaluation側の仕様に合わせて new_columns は変更しない）
train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109',
               'U0111', 'U0201', 'U0202', 'U0203', 'U0204',
               'U0206', 'U0207', 'U0208', 'U0210']
val_users = ['U0101', 'U0209']
test_users = ['U0105', 'U0110', 'U0205', 'U0209']

selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'operation','action',"timestamp"]


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
# 各 (operation, action) のセグメント長分布を計算
# ---------------------------
def calculate_segment_length_distributions(train_data_dict):
    """
    Calculate segment length distributions for each (operation, action) pair.
    
    Args:
        train_data_dict (dict): Dictionary mapping user IDs to their dataframes
        
    Returns:
        tuple: (segment_length_dist, raw_segment_length_dist) where:
            - segment_length_dist: Dict of filtered segment lengths by (operation, action)
            - raw_segment_length_dist: Dict of raw segment lengths by (operation, action)
    """
    segment_length_dist = {}  # キーは (operation, action) タプル
    raw_segment_length_dist = {}  # 外れ値除去前の元データも保存しておく

    for u, df in train_data_dict.items():
        df_temp = df.copy()
        # operationまたはactionが変化したら新しいグループとする
        df_temp['group'] = ((df_temp['operation'] != df_temp['operation'].shift()) | 
                            (df_temp['action'] != df_temp['action'].shift())).cumsum()
        for _, group_df in df_temp.groupby('group'):
            key = (group_df['operation'].iloc[0], group_df['action'].iloc[0])
            seg_len = len(group_df)
            raw_segment_length_dist.setdefault(key, []).append(seg_len)

    # IQRを使用して外れ値を除外する
    for key, lengths in raw_segment_length_dist.items():
        lengths_arr = np.array(lengths)
        
        # IQR計算
        q1 = np.percentile(lengths_arr, 25)
        q3 = np.percentile(lengths_arr, 75)
        iqr = q3 - q1
        
        # 外れ値の境界を定義 (一般的に使われる1.5*IQRを使用)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 外れ値でないデータのみをフィルタリング
        filtered_lengths = lengths_arr[(lengths_arr >= lower_bound) & (lengths_arr <= upper_bound)]
        
        # 外れ値を除いたデータを保存
        segment_length_dist[key] = filtered_lengths.tolist()

    # 各セグメントの統計量を出力（デバッグ用）
    for key, lengths in segment_length_dist.items():
        lengths_arr = np.array(lengths)
        raw_lengths = np.array(raw_segment_length_dist[key])
        
        # フィルタリング前後の統計情報を出力
        print(f"Operation {key[0]}, Action {key[1]}: ")
        print(f"  Raw: count={len(raw_lengths)}, mean={raw_lengths.mean():.2f}, median={np.median(raw_lengths):.2f}, std={raw_lengths.std():.2f}")
        print(f"  Filtered: count={len(lengths_arr)}, mean={lengths_arr.mean():.2f}, median={np.median(lengths_arr):.2f}, std={lengths_arr.std():.2f}")
        print(f"  Removed: {len(raw_lengths) - len(lengths_arr)} outliers")
    
    return segment_length_dist, raw_segment_length_dist

# Call the function to calculate segment length distributions
segment_length_dist, raw_segment_length_dist = calculate_segment_length_distributions(train_data_dict)

# ---------------------------
# 補助関数の定義
# ---------------------------
def generate_unique_filename(prefix):
    """ユーザID等を接頭辞にしてユニークなファイル名を生成する"""
    return f"{prefix}_{uuid.uuid4().hex}.csv"
def delete_csv_files(folder_path):
    """
    Deletes all CSV files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for file_path in csv_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Skip broken symlinks
                total_size += os.path.getsize(fp)
    return total_size

def check_virtual_volume(rootdir, virtpath, limit_mb=500):
    """
    Check if the total size of the virtual data folder stays within limit_mb megabytes.
    
    Parameters:
      rootdir (str): The base root directory.
      virtpath (str): The virtual data folder path relative to rootdir.
      limit_mb (int): The size limit in MB (default is 500 MB).
    """
    virt_directory = rootdir + virtpath
    total_virtual_size = get_folder_size(virt_directory)
    limit = limit_mb * 1024 * 1024  # convert MB to bytes

    if total_virtual_size <= limit:
        # print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、{limit_mb}MB以下です。")
        return True
    else:
        # print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、{limit_mb}MBを超えています。")
        return False
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
    threshold = np.median(np.abs(coeff[-1])) / 0.6745
    new_coeff = [pywt.threshold(c, threshold, mode='soft') for c in coeff]
    rec = pywt.waverec(new_coeff, wavelet)
    rec = rec[:len(series)]
    return pd.Series(rec, index=series.index)

def filter_kalman(series):
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
    state_means, _ = kf.smooth(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def filter_savgol(series, window_length=7, polyorder=2):
    if window_length >= len(series):
        window_length = len(series) if len(series) % 2 == 1 else len(series) - 1
    # Ensure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
    return pd.Series(savgol_filter(series, window_length=window_length, polyorder=polyorder), index=series.index)

def apply_outlier_filter(series, method='none'):
    if method == 'moving_average':
        return filter_moving_average(series)
    elif method == 'ransac':
        return filter_ransac(series)
    elif method == 'wavelet':
        return filter_wavelet(series)
    elif method == 'kalman':
        return filter_kalman(series)
    elif method == 'savgol':
        return filter_savgol(series)
    elif method == 'none':
        return series
    else:
        return series

# --- 補間手法の定義 ---
def interpolate_linear(x, y, new_x):
    # Replace non-finite values in y with linearly interpolated values if any
    if not np.all(np.isfinite(y)):
        y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(new_x)

def interpolate_rbf(x, y, new_x, function='multiquadric'):
    # Replace non-finite values in y with linearly interpolated values if any
    if not np.all(np.isfinite(y)):
        y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
    rbf_func = Rbf(x, y, function=function)
    return rbf_func(new_x)

def interpolate_bspline(x, y, new_x, k=3, s=0):
    if len(x) <= k:
        return np.interp(new_x, x, y)
    tck = splrep(x, y, k=k, s=s)
    return splev(new_x, tck)

def interpolate_akima(x, y, new_x):
    # Replace non-finite values in y with linearly interpolated values if any
    if not np.all(np.isfinite(y)):
        y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
    try:
        akima = Akima1DInterpolator(x, y)
        return akima(new_x)
    except ValueError:
        return np.interp(new_x, x, y)

def interpolate_pchip(x, y, new_x):
    # Replace non-finite values in y with linearly interpolated values if any
    if not np.all(np.isfinite(y)):
        y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
    pchip = PchipInterpolator(x, y)
    return pchip(new_x)

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
    Interpolate missing values (NaNs) in a time series using a hybrid approach:
    - Linear interpolation for gaps shorter or equal to max_linear_gap.
    - Cubic spline interpolation for larger gaps.
    After filling missing values on the original time axis, the result is interpolated onto new_x.
    """
    values = np.asarray(values).copy()
    n = len(values)
    isnan = np.isnan(values)
    if not np.any(isnan):
        if len(new_x) != n:
            return np.interp(new_x, time, values)
        return values
    
    # Find gaps (continuous NaN segments) by scanning through data
    i = 0
    while i < n:
        if np.isnan(values[i]):
            start = i
            while i < n and np.isnan(values[i]):
                i += 1
            end = i
            gap_length = end - start
            if gap_length <= max_linear_gap:
                if start == 0 or end == n:
                    continue
                x0, x1 = time[start-1], time[end]
                y0, y1 = values[start-1], values[end]
                interp_times = time[start:end]
                values[start:end] = y0 + (y1 - y0) * ((interp_times - x0) / (x1 - x0))
            else:
                if start == 0 or end == n:
                    if start == 0 and end < n:
                        values[start:end] = values[end]
                    elif end == n and start > 0:
                        values[start:end] = values[start-1]
                    continue
                idx_before = start - 1
                idx_after = end
                spline_x = time[[idx_before, idx_after]]
                spline_y = values[[idx_before, idx_after]]
                cs = CubicSpline(spline_x, spline_y, bc_type='natural')
                interp_times = time[start:end]
                values[start:end] = cs(interp_times)
        else:
            i += 1
    if len(new_x) != n:
        values = np.interp(new_x, time, values)
    return values
import numpy as np
from scipy.signal.windows import blackmanharris
from scipy.special import gamma
def interpolate_fft(x, y, new_x, window='hann'):
    """
    FFTベースの補間を実施する関数

    Parameters:
      x : array-like
          元のタイムスタンプ（均等サンプリングされている前提）
      y : array-like
          補間対象のデータ系列
      new_x : array-like
          補間後のタイムスタンプ。補間結果は新たに一様な系列として得られるが、
          new_xが元の一様系列と異なる場合、線形補間により最終的な値を算出する。
      window : str, optional
          使用する窓関数。'hann'、'Welch'、'Blackman-Harris'、もしくはその他（デフォルトは'hann'）。
    
    Returns:
      y_final : array-like
          new_xに対応する補間後のデータ系列
    """
    # numpy配列に変換
    x = np.asarray(x)
    y = np.asarray(y)
    new_x = np.asarray(new_x)
    
    N = len(y)
    new_N = len(new_x)
    
    # --- 窓関数の生成 ---
    if window == 'hann':
        win = np.hanning(N)
    elif window == 'Welch':
        # Welch窓はパラボリック窓とも呼ばれる
        n = np.arange(N)
        win = 1 - ((n - (N-1)/2) / ((N-1)/2))**2
    elif window == 'Blackman-Harris':
        win = blackmanharris(N)
    else:
        win = np.ones(N)
    
    # 入力系列に窓を適用
    y_windowed = y * win

    # --- FFTを計算 ---
    Y = np.fft.fft(y_windowed)
    
    # --- ゼロパディングまたはトランケーション ---
    if new_N > N:
        pad = new_N - N
        # FFT係数の左右対称性を保つため、中央にゼロを挿入
        if N % 2 == 0:
            left = Y[:N//2]
            right = Y[N//2:]
        else:
            left = Y[:(N+1)//2]
            right = Y[(N+1)//2:]
        Y_padded = np.concatenate([left, np.zeros(pad, dtype=complex), right])
    else:
        # 補間先のサイズが小さい場合は先頭new_N個を採用（必要に応じた実装変更も可）
        Y_padded = Y[:new_N]
    
    # --- 逆FFTにより時系列信号に戻す ---
    y_interp = np.fft.ifft(Y_padded)
    # 実部を取り、スケーリング（新旧データ数の比率）で調整
    y_interp = np.real(y_interp) * (float(new_N) / N)
    
    # --- FFT補間結果は一様なタイム軸上のデータとなる ---
    t_uniform = np.linspace(x[0], x[-1], new_N)
    
    # new_xが一様なタイム軸と一致しない場合は、線形補間で最終調整
    if not np.allclose(new_x, t_uniform):
        y_final = np.interp(new_x, t_uniform, y_interp)
    else:
        y_final = y_interp

    return y_final


def dtw_align_paths(seq1, seq2, dist=lambda x, y: abs(x - y)):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    ptr = np.zeros((n+1, m+1, 2), dtype=int)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(seq1[i-1], seq2[j-1])
            choices = [dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]]
            idx = np.argmin(choices)
            dtw[i, j] = cost + choices[idx]
            if idx == 0:
                ptr[i, j] = [i-1, j]
            elif idx == 1:
                ptr[i, j] = [i, j-1]
            else:
                ptr[i, j] = [i-1, j-1]
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        i_prev, j_prev = ptr[i, j]
        i, j = i_prev, j_prev
    path.reverse()
    return path

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
      1. train_data_dict内の各CSVデータについて、operationおよびactionの変化ごとにセグメントを抽出
         （セグメント長分布は事前に計算した分布を利用）
      2. 各セグメントのtimestampを先頭0起点に調整
      3. 指定の外れ値除去法で加速度データをフィルタリング
      4. 指定の補間手法で各セグメントを補間し、新しいタイムスタンプ系列にリサンプリング
         - num_timestamps=Trueの場合、各セグメントの長さを、(operation, action)ごとの実データ分布からランダムにサンプリングする
         - num_timestamps=Noneの場合は元のtimestamp系列をそのまま利用
      5. 必要に応じて値を丸め（round_values）
      6. CSVとして出力し、出力ディレクトリの容量がlimit_mb未満なら処理を継続
    
    Args:
        distribution_type: 生成するセグメント長の分布タイプ ('normal', 'gamma', 'exponential')
    """
    # 出力先内のCSVをすべて削除
    calculate_segment_length_distributions(train_data_dict)
    delete_csv_files(output_dir)
    rootdir = '/root/Virtual_Data_Generation'
    virtpath = '/data/virtual'
    # ループ：出力ディレクトリ容量が500MB未満の場合に生成（容量超えたら終了）
    while check_virtual_volume(rootdir, virtpath, limit_mb=500) is True:
        for user, df in train_data_dict.items():
            if check_virtual_volume(rootdir, virtpath, limit_mb=500) is False:
                return None
            df_proc = df[selected_columns].copy()
            # operationまたはactionが変化したら新グループとする
            df_proc['group'] = ((df_proc['operation'] != df_proc['operation'].shift()) |
                                (df_proc['action'] != df_proc['action'].shift())).cumsum()
            processed_groups = []
            for group_id, group_df in df_proc.groupby('group'):
                group_df = group_df.copy()
                # タイムスタンプを先頭0起点に調整
                group_df['timestamp'] = group_df['timestamp'] - group_df['timestamp'].iloc[0]
                
                # 新しいtimestamp系列の生成
                if num_timestamps:
                    op_val = group_df['operation'].iloc[0]
                    act_val = group_df['action'].iloc[0]
                    key = (op_val, act_val)
                    if key in segment_length_dist:
                        durations = np.array(segment_length_dist[key])
                        mean_val = durations.mean()
                        std_val = durations.std()
                        
                        # 選択された分布タイプに基づいてサンプリング
                        if distribution_type == 'normal':
                            current_num = int(np.round(np.random.normal(mean_val, std_val)))
                        elif distribution_type == 'gamma':
                            # ガンマ分布パラメータ: シェイプ k とスケール θ を推定
                            # 正規分布の平均と分散からガンマ分布のパラメータを推定
                            if mean_val > 0 and std_val > 0:
                                shape = (mean_val / std_val) ** 2
                                scale = std_val ** 2 / mean_val
                                current_num = int(np.round(np.random.gamma(shape, scale)))
                            else:
                                current_num = int(mean_val)
                        elif distribution_type == 'exponential':
                            # 指数分布のパラメータ λ = 1/平均
                            if mean_val > 0:
                                scale = mean_val  # 指数分布の scale=1/rate
                                current_num = int(np.round(np.random.exponential(scale)))
                            else:
                                current_num = int(mean_val)
                        elif distribution_type == 'lognormal':
                            # 対数正規分布のパラメータ μ と σ を推定
                            if mean_val > 0 and std_val > 0:
                                var = std_val ** 2
                                mu = np.log(mean_val**2 / np.sqrt(var + mean_val**2))
                                sigma = np.sqrt(np.log(1 + var / mean_val**2))
                                current_num = int(np.round(np.random.lognormal(mu, sigma)))
                            else:
                                current_num = int(mean_val)
                        elif distribution_type == 'weibull':
                            # ワイブル分布のパラメータ k と λ を推定
                            # 形状パラメータ k は1.2（やや右に裾が長い）、スケールパラメータはmean_valを基準に調整
                            if mean_val > 0:
                                k = 1.2  # 形状パラメータ
                                # Γ(1 + 1/k) = ガンマ関数
                                lambda_param = mean_val / gamma(1 + 1/k)
                                current_num = int(np.round(np.random.weibull(k) * lambda_param))
                            else:
                                current_num = int(mean_val)
                        elif distribution_type == 'poisson':
                            # ポアソン分布（整数値を生成する）
                            if mean_val > 0:
                                current_num = np.random.poisson(mean_val)
                            else:
                                current_num = 1
                        else:
                            # デフォルトは正規分布
                            current_num = int(np.round(np.random.normal(mean_val, std_val)))
                            
                        current_num = max(current_num, 1)
                    else:
                        current_num = max(len(group_df), 1)
                    new_timestamps = np.linspace(group_df['timestamp'].min(),
                                                 group_df['timestamp'].max(),
                                                 current_num)
                else:
                    new_timestamps = group_df['timestamp'].values
                
                # 新しいグループDataFrameを作成（補間後の行数 = new_timestampsの長さ）
                new_group_df = pd.DataFrame({'timestamp': new_timestamps})
                # グループ内の行数が1なら、外れ値フィルタや補間処理をスキップ
                if len(group_df) == 1:
                    if num_timestamps is not None:
                        # num_timestamps が指定されていれば、新しいDataFrameを作成する
                        ts_min = group_df['timestamp'].min()
                        ts_max = group_df['timestamp'].max()
                        new_timestamps = np.linspace(ts_min, ts_max, num_timestamps)
                        # 元のデータを複製して新しいDataFrameを作成
                        new_df = pd.DataFrame([group_df.iloc[0].to_dict()] * num_timestamps)
                        new_df['timestamp'] = new_timestamps
                        processed_groups.append(new_df)
                    else:
                        processed_groups.append(group_df)
                    continue
                # 各説明変数に対してフィルタと補間を実施
                for col in explanatory_columns:
                    filtered_series = apply_outlier_filter(group_df[col], method=outlier_method)
                    interp_values = perform_interpolation(group_df['timestamp'].values,
                                                          filtered_series.values,
                                                          new_timestamps,
                                                          method=interpolation_method)
                    new_group_df[col] = interp_values
                # operation列を付与（actionは評価には使わないので省略可）
                new_group_df['operation'] = group_df['operation'].iloc[0]
                processed_groups.append(new_group_df)
            final_df = pd.concat(processed_groups, ignore_index=True)
            if round_values:
                final_df = final_df.round(5)
            if 'timestamp' in final_df.columns:
                final_df = final_df.drop(columns=['timestamp'])
            if 'action' in final_df.columns:
                final_df = final_df.drop(columns=['action'])
            if 'group' in final_df.columns:
                final_df = final_df.drop(columns=['group'])
            output_file = os.path.join(output_dir, generate_unique_filename(user))
            final_df.to_csv(output_file, index=False)
            if check_virtual_volume(rootdir, virtpath, limit_mb=500) is False:
                return None

# ---------------------------
# 実行例
# ---------------------------
if __name__ == '__main__':
    f1_scores = {}
    distribution_types = ['normal']
    for dist_type in distribution_types:
        f1_scores[dist_type] = []
        for outlier_method in ["none"]:
            for interp_method in ["rbf_multiquadric"]:
                f1_scores_runs = []
                for run in range(1):  # Run each experiment 3 times
                    process_csv_files(output_dir='/root/Virtual_Data_Generation/data/virtual',
                                     num_timestamps=100,  # Using a fixed value to test distribution effects
                                     outlier_method=outlier_method,
                                     interpolation_method=interp_method,
                                     round_values=True,
                                     distribution_type=dist_type)
    print("All processing done.")