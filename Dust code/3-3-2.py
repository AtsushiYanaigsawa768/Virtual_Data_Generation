import os
import uuid
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf, BarycentricInterpolator, KroghInterpolator, PchipInterpolator, Akima1DInterpolator
from scipy.interpolate import lagrange, BSpline, splrep, splev
from scipy.signal import savgol_filter
import pywt
from pykalman import KalmanFilter
from sklearn.linear_model import RANSACRegressor
from evaluation import HAR_evaluation
from delete import delete_csv_files
from volumeChecker import check_virtual_volume
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
                    'operation',"timestamp","action"]

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
    new_coeff = [pywt.threshold(c, threshold, mode='soft') for c in coeff]
    rec = pywt.waverec(new_coeff, wavelet)
    rec = rec[:len(series)]
    return pd.Series(rec, index=series.index)

def filter_kalman(series):
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
    state_means, _ = kf.smooth(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def filter_savgol(series, window_length=7, polyorder=2):
    # window_lengthは系列長以下の奇数にする
    if window_length >= len(series):
        window_length = len(series) if len(series) % 2 == 1 else len(series) - 1
    return pd.Series(savgol_filter(series, window_length=window_length, polyorder=polyorder), index=series.index)

def apply_outlier_filter(series, method='moving_average'):
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

# --- 補間処理 ---
def interpolate_linear(x, y, new_x):
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(new_x)

def interpolate_rbf(x, y, new_x, function='multiquadric'):
    rbf_func = Rbf(x, y, function=function)
    return rbf_func(new_x)

def interpolate_bspline(x, y, new_x, k=3, s=0):
    tck = splrep(x, y, k=k, s=s)
    return splev(new_x, tck)

def interpolate_gpr(x, y, new_x):
    from sklearn.gaussian_process import GaussianProcessRegressor
    x = np.array(x).reshape(-1, 1)
    new_x = np.array(new_x).reshape(-1, 1)
    gpr = GaussianProcessRegressor().fit(x, y)
    return gpr.predict(new_x)

import numpy as np
from scipy.signal.windows import blackmanharris

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


def interpolate_lagrange(x, y, new_x):
    poly = lagrange(x, y)
    return poly(new_x)

def interpolate_barycentric(x, y, new_x):
    interpolator = BarycentricInterpolator(x, y)
    return interpolator(new_x)

def interpolate_akima(x, y, new_x):
    akima = Akima1DInterpolator(x, y)
    return akima(new_x)

def interpolate_pchip(x, y, new_x):
    pchip = PchipInterpolator(x, y)
    return pchip(new_x)

def perform_interpolation(x, y, new_x, method='pchip', **kwargs):
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
    # elif method == 'gpr': time consuming but accurate
    #     return interpolate_gpr(x, y, new_x)
    elif method == 'fft_hann':
        return interpolate_fft(x, y, new_x, window='hann')
    elif method == 'fft_Welch':
        return interpolate_fft(x, y, new_x, window='Welch')
    elif method == 'fft_Blackman-Harris':
        return interpolate_fft(x, y, new_x, window='Blackman-Harris')
    # elif method == 'lagrange':
    #     return interpolate_lagrange(x, y, new_x)
    # elif method == 'barycentric':
    #     return interpolate_barycentric(x, y, new_x)
    # elif method == 'akima':
    #     return interpolate_akima(x, y, new_x)
    elif method == 'pchip':
        return interpolate_pchip(x, y, new_x)
    else:
        return interpolate_linear(x, y, new_x)

# ---------------------------
# メイン処理関数
# ---------------------------
def process_csv_files(output_dir, num_timestamps=None,
                      outlier_method='moving_average', interpolation_method='pchip',
                      round_values=False):
    """
    output_dir を引数に取り、まず出力先内のCSVファイルを削除。
    その後、train_data_dict内の各CSVファイルについて以下の処理を行う。
      1. 指定された列のみ抽出し、操作(operation)とactionの変化毎にgroupを作成
      2. 各groupについて、先頭のtimestampを0に調整
      3. 各説明変数(explanatory_columns)に対して、指定の外れ値フィルタリングを実施
      4. その後、指定の補間手法により補間処理を実施
         ※timestampの補間は、（1）調整後の元の値そのままか、（2）最初の値0と最大値をnum_timestamps個に均等割りした値かのどちらか
      5. 全groupの処理終了後、必要に応じて数値を少数第5位で四捨五入
      6. ユニークな名前でCSV出力
      7. 出力後、出力ディレクトリの容量が limit_mb 未満かをチェックし、
         条件を満たすまでループ処理を継続
      8. さらに、num_timestamps が None の場合、
         1ループが完了した後、すでに生成されている各ユーザのCSVファイルの中身を複製して
         別のユニークな名前で保存する。
    """
    # 出力先内のCSVをすべて削除
    delete_csv_files(output_dir)
    rootdir = '/root/Virtual_Data_Generation'
    virtpath = '/data/virtual'
    
    # 第一段階：通常の処理ループ
    while check_virtual_volume(rootdir, virtpath, limit_mb=500) is True:
        for user, df in train_data_dict.items():
            # dfのコピーを作成し、指定列に絞る
            df_proc = df[selected_columns].copy()
            # operation と action の変化毎にグループを作成
            df_proc['group'] = ((df['operation'] != df['operation'].shift()) | 
                                (df['action'] != df['action'].shift())).cumsum()
            
            processed_groups = []
            # 各 group ごとに処理
            for group_id, group_df in df_proc.groupby('group'):
                group_df = group_df.copy()
                # 各 group の timestamp の最初の値を0に調整
                group_df['timestamp'] = group_df['timestamp'] - group_df['timestamp'].iloc[0]
                
                # グループ内の行数が1なら、外れ値フィルタや補間処理をスキップ
                if len(group_df) == 1:
                    if num_timestamps is not None:
                        # num_timestamps が指定されていれば、timestamp のみ均等割りにする
                        group_df['timestamp'] = np.linspace(group_df['timestamp'].min(),
                                                            group_df['timestamp'].max(),
                                                            num_timestamps)
                    processed_groups.append(group_df)
                    continue

                # 補間用の新しい timestamp の生成
                if num_timestamps is not None:
                    new_timestamps = np.linspace(group_df['timestamp'].min(),
                                                 group_df['timestamp'].max(),
                                                 num_timestamps)
                else:
                    new_timestamps = group_df['timestamp'].values

                # 各説明変数に対して、外れ値フィルタと補間を実施
                for col in explanatory_columns:
                    # 外れ値フィルタリング
                    filtered_series = apply_outlier_filter(group_df[col], method=outlier_method)
                    # 補間（x: 調整済 timestamp, y: フィルタ済データ）
                    interp_values = perform_interpolation(group_df['timestamp'].values,
                                                          filtered_series.values,
                                                          new_timestamps,
                                                          method=interpolation_method)
                    # 補間結果を更新（補間結果の長さが new_timestamps と一致することを前提）
                    group_df.loc[:, col] = interp_values

                # num_timestamps 指定時は timestamp を新たな値に置換
                if num_timestamps is not None:
                    group_df['timestamp'] = new_timestamps
                    
                processed_groups.append(group_df)

            final_df = pd.concat(processed_groups, ignore_index=True)
            
            # round_values が True の場合、数値を少数第5位で四捨五入
            if round_values:
                final_df = final_df.round(5)
            
            # 不要な列を削除
            final_df = final_df.drop(columns=['group', 'timestamp', 'action'])
            
            # ユニークなファイル名で出力
            output_file = os.path.join(output_dir, generate_unique_filename(user))
            final_df.to_csv(output_file, index=False)
            
            # 出力後、容量チェック。条件を満たさなくなったら終了
            if check_virtual_volume(rootdir, virtpath, limit_mb=500) is False:
                return None

        # 第二段階：num_timestamps が None の場合、生成済み CSV を複製する
        if num_timestamps is None:
            while check_virtual_volume(rootdir, virtpath, limit_mb=500) is True:
                # 各ユーザに対して、出力ディレクトリ内でそのユーザに対応する CSV を探す
                for user in train_data_dict.keys():
                    for filename in os.listdir(output_dir):
                        if filename.startswith(f"{user}_") and filename.endswith(".csv"):
                            filepath = os.path.join(output_dir, filename)
                            # CSV ファイルの内容を読み込む
                            df_dup = pd.read_csv(filepath)
                            # 新たなユニークなファイル名で複製して保存
                            new_filepath = os.path.join(output_dir, generate_unique_filename(user))
                            df_dup.to_csv(new_filepath, index=False)
                            # 複製後、容量チェック
                            if check_virtual_volume(rootdir, virtpath, limit_mb=500) is False:
                                return None


# ---------------------------
# 実行例
# ---------------------------
if __name__ == '__main__':
    # Experiment 1: No outlier filtering, linear interpolation, no rounding
    f1_scores = []
    for outlier_method in ['none']:
        for interp_method in ["interpolate", 'rbf_inverse', 'rbf_multiquadric', 'rbf_gaussian', 'rbf_linear', 'rbf_cubic', 'fft_hann', 'fft_Welch', 'fft_Blackman-Harris','pchip', 'akima', ]:
            process_csv_files(output_dir='/root/Virtual_Data_Generation/data/virtual',
                                   num_timestamps=None,
                                   outlier_method=outlier_method,
                                   interpolation_method=interp_method,
                                   round_values=False)
            f1 = HAR_evaluation(f"Action_outlier_{outlier_method}_interp_{interp_method}_False",'/root/Virtual_Data_Generation/data/virtual')
            print(f"Outlier method: {outlier_method}, Interpolation method: {interp_method} -> f1 score: {f1}")
            f1_scores.append(f1)
    print(f1_scores)
    print("All processing done.")
    # Experiment 2: No outlier filtering, linear interpolation, rounding
    f1_scores = []
    for outlier_method in ['none']:
        for interp_method in ["interpolate",'rbf_inverse', 'rbf_multiquadric', 'rbf_gaussian', 'rbf_linear', 'rbf_cubic', 'fft_hann', 'fft_Welch', 'fft_Blackman-Harris','pchip', 'akima',]:
            process_csv_files(output_dir='/root/Virtual_Data_Generation/data/virtual',
                                   num_timestamps=None,
                                   outlier_method=outlier_method,
                                   interpolation_method=interp_method,
                                   round_values=True)
            f1 = HAR_evaluation(f"Action_outlier_{outlier_method}_interp_{interp_method}_True",'/root/Virtual_Data_Generation/data/virtual')
            print(f"Outlier method: {outlier_method}, Interpolation method: {interp_method} -> f1 score: {f1}")
            f1_scores.append(f1)
    print(f1_scores)
    print("All processing done.")
    # Experiment 3: Moving average outlier filtering, linear interpolation, no rounding
    # f1_scores = []
    # for outlier_method in [  'wavelet','kalman',]:
    #     for interp_method in [ "bspline",  'rbf_linear',  'fft_Blackman-Harris', ]:
    #         if "bspline" == interp_method or "fft_Blackman-Harris" == interp_method:
    #             process_csv_files(output_dir='/root/Virtual_Data_Generation/data/virtual',
    #                                 num_timestamps=None,
    #                                 outlier_method=outlier_method,
    #                                 interpolation_method=interp_method,
    #                                 round_values=False)
    #         else:
    #             process_csv_files(output_dir='/root/Virtual_Data_Generation/data/virtual',
    #                                 num_timestamps=None,
    #                                 outlier_method=outlier_method,
    #                                 interpolation_method=interp_method,
    #                                 round_values=True)
    #         f1 = HAR_evaluation(f"3_outlier_{outlier_method}_interp_{interp_method}",'/root/Virtual_Data_Generation/data/virtual')
    #         print(f"Outlier method: {outlier_method}, Interpolation method: {interp_method} -> f1 score: {f1}")
    #         f1_scores.append(f1)
