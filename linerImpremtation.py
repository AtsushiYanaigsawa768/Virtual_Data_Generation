import os
import glob
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d, Rbf, make_interp_spline
from scipy.optimize import curve_fit
from evaluation import HAR_evaluation
from delete import delete_csv_files
from volumeChecker import check_virtual_volume
from scipy.signal.windows import blackmanharris, hann
from pykalman import KalmanFilter
import glob
import pandas as pd
import numpy as np
import random

# --- Globals and call example ---
realpath = r'/data/real'
virtpath = r'/data/test'
rootdir = r'/root/Virtual_Data_Generation'
real_directory = os.path.join(rootdir, realpath)
virt_directory = os.path.join(rootdir, virtpath)

# --- Noise reduction functions ---
def noise_reduction_wavelet(df, signal_columns, wavelet='db1', level=1):
    """
    PyWavelets を用いて各信号列のノイズ除去を行う。
    詳細係数にソフト閾値処理を施し、再構成する。
    """
    import pywt
    import numpy as np
    df_denoised = df.copy()
    for col in signal_columns:
        signal = df[col].values
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # 最後の詳細係数から閾値を算出
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised_signal = pywt.waverec(coeffs, wavelet)
        # 再構成後の長さが元と異なる場合の補正
        denoised_signal = denoised_signal[:len(signal)]
        df_denoised[col] = denoised_signal
    return df_denoised


def noise_reduction_kalman(df, signal_columns):
    """
    pykalman を用いて各信号列のスムージング(ノイズ除去)を行う。
    """
    df_denoised = df.copy()
    for col in signal_columns:
        signal = df[col].values
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=signal[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )
        state_means, _ = kf.smooth(signal)
        df_denoised[col] = state_means.flatten()
    return df_denoised


def noise_reduction_savgol(df, signal_columns, window_length=5, polyorder=2):
    """
    SciPy の savgol_filter を用いて各信号列のノイズ抑制を行う。
    """
    from scipy.signal import savgol_filter
    import numpy as np
    df_denoised = df.copy()
    for col in signal_columns:
        signal = df[col].values
        # window_length は奇数でかつ信号の長さ以下である必要がある
        wl = window_length
        if len(signal) < wl:
            wl = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
            if wl < 3:
                wl = 3
        if wl % 2 == 0:  # 奇数になるよう調整
            wl += 1
        df_denoised[col] = savgol_filter(signal, wl, polyorder)
    return df_denoised
def remove_outliers(df_grouped, method='moving_average'):
    df_clean = df_grouped.copy()
    non_timestamp = [col for col in df_grouped.columns if col != 'timestamp']
    if method == 'moving_average':
        for col in non_timestamp:
            # Apply rolling mean (centered) then fill initial/final NaNs.
            df_clean[col] = df_clean[col].rolling(window=3, center=True).mean()
            df_clean[col] = df_clean[col].bfill().ffill()
    elif method == 'ransac':
        from sklearn.linear_model import RANSACRegressor
        for col in non_timestamp:
            X = df_clean['timestamp'].values.reshape(-1, 1)
            y = df_clean[col].values
            model = RANSACRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)
            df_clean[col] = y_pred
    return df_clean

# --- Interpolation functions ---
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

def rbf_adapt(random_times, df_grouped, function='multiquadric', scale_factor=1.0):
    results = []
    ts_array = df_grouped['timestamp'].values
    std_ts = np.std(ts_array) if np.std(ts_array) != 0 else 1.0
    epsilon = std_ts * scale_factor
    rbfs = {}
    for col in df_grouped.columns:
        if col == "timestamp":
            continue
        rbfs[col] = Rbf(ts_array, df_grouped[col].values, function=function, epsilon=epsilon)
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

def bspline_interpolation(random_times, df_grouped, degree=5):
    results = []
    ts_array = df_grouped['timestamp'].values
    non_timestamp = [col for col in df_grouped.columns if col != "timestamp"]
    for rt in random_times:
        row = {"timestamp": rt}
        for col in non_timestamp:
            y = df_grouped[col].values
            spline = make_interp_spline(ts_array, y, k=degree)
            row[col] = float(spline(rt))
        results.append(row)
    return results

def gpr_interpolation(random_times, df_grouped):
    results = [{"timestamp": rt} for rt in random_times]
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    ts_array = df_grouped['timestamp'].values.reshape(-1, 1)
    non_timestamp = [col for col in df_grouped.columns if col != "timestamp"]
    for col in non_timestamp:
        y = df_grouped[col].values
        # Kernel: RBF with a white noise kernel added.
        kernel = RBF(length_scale=np.std(ts_array)) + WhiteKernel(noise_level=1)
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(ts_array, y)
        preds = gpr.predict(np.array(random_times).reshape(-1, 1))
        for idx, pred in enumerate(preds):
            results[idx][col] = float(pred)
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

def welch_window(M):
    n = np.arange(M)
    return 1 - ((n - (M - 1) / 2) / ((M - 1) / 2))**2

def fft_interpolate(fs, amp_array, time_array, k=0, window='Welch'):
    N = len(amp_array)
    # Choose a window: Welch, Blackman-Harris, or Hann.
    if window.lower() == 'welch':
        win = welch_window(N)
    elif window.lower() == 'blackman-harris':
        win = blackmanharris(N)
    elif window.lower() == 'hann':
        win = hann(N)
    else:
        win = np.ones(N)
    amp_array = amp_array * win

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

def fft_implementation(random_times, df_grouped, k=0, window_choice='Welch'):
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
        interpolated_vals = fft_interpolate(fs, amp_array, rt_array, k=k, window=window_choice)
        for idx, val in enumerate(interpolated_vals):
            results[idx][col] = float(val)
    return results

def linear_high_performance(random_times, df_grouped):
    results = []
    ts_array = df_grouped['timestamp'].values
    non_timestamp_cols = [col for col in df_grouped.columns if col != "timestamp"]
    
    interpolated_data = {"timestamp": np.array(random_times)}
    for col in non_timestamp_cols:
        interpolated_data[col] = np.interp(random_times, ts_array, df_grouped[col].values)

    for idx in range(len(random_times)):
        row = {key: float(interpolated_data[key][idx]) if key != "timestamp" else interpolated_data[key][idx] 
               for key in interpolated_data}
        results.append(row)
    return results

def linear_precision(random_times, df_grouped):
    results = []
    ts_array = df_grouped['timestamp'].values.astype(np.float64)
    non_timestamp_cols = [col for col in df_grouped.columns if col != "timestamp"]

    for rt in random_times:
        row = {"timestamp": rt}
        for col in non_timestamp_cols:
            pos = np.searchsorted(ts_array, rt)
            if pos == 0:
                t0, t1 = ts_array[0], ts_array[1]
                y0, y1 = df_grouped.iloc[0][col], df_grouped.iloc[1][col]
            elif pos >= len(ts_array):
                t0, t1 = ts_array[-2], ts_array[-1]
                y0, y1 = df_grouped.iloc[-2][col], df_grouped.iloc[-1][col]
            else:
                t0, t1 = ts_array[pos - 1], ts_array[pos]
                y0, y1 = df_grouped.iloc[pos - 1][col], df_grouped.iloc[pos][col]
            ratio = (rt - t0) / (t1 - t0)
            try:
                interp_val = math.fma((y1 - y0), ratio, y0)
            except AttributeError:
                interp_val = y0 + (y1 - y0) * ratio
            row[col] = interp_val
        results.append(row)
    return results

# --- Main interpolation routine ---
# --- New Interpolation using ARIMA ---
def arima_interpolation(random_times, df_grouped, order=(1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    results = []
    ts = df_grouped['timestamp'].values
    non_timestamp = [col for col in df_grouped.columns if col != "timestamp"]
    predictions = {}
    # Fit an ARIMA model for every column, predict on the full index.
    for col in non_timestamp:
        series = df_grouped[col].values
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            pred_series = model_fit.predict(start=0, end=len(series)-1)
        except Exception:
            pred_series = series  # fallback if model fails
        predictions[col] = pred_series
    # Map random_times to indices by a linear scaling over the time range.
    indices = np.linspace(0, len(ts)-1, len(ts))
    for rt in random_times:
        # Compute fractional index corresponding to the timestamp.
        idx_f = ((rt - ts[0]) / (ts[-1] - ts[0])) * (len(ts) - 1) if ts[-1] != ts[0] else 0
        row = {"timestamp": rt}
        for col in non_timestamp:
            interp_val = np.interp(idx_f, indices, predictions[col])
            row[col] = float(interp_val)
        results.append(row)
    return results

# # --- New Interpolation using Bayesian Inference (with pymc3) ---
# def bayesian_interpolation(random_times, df_grouped):
#     results = []
#     ts = df_grouped['timestamp'].values
#     non_timestamp = [col for col in df_grouped.columns if col != "timestamp"]
#     predictions = {}
#     # For each column, we assume a simple linear trend and sample its posterior.
#     for col in non_timestamp:
#         y = df_grouped[col].values
#         with pm.Model() as model:
#             intercept = pm.Normal('intercept', mu=0, sigma=10)
#             slope = pm.Normal('slope', mu=0, sigma=10)
#             sigma = pm.HalfNormal('sigma', sigma=1)
#             mu = intercept + slope * ts
#             y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
#             trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
#         # Use the posterior means to form a trend.
#             pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
#         slope_mean = trace['slope'].mean()
#         predicted = intercept_mean + slope_mean * ts
#         predictions[col] = predicted
#     # For new timestamps, interpolate using the already computed predictions.
#     for rt in random_times:
#         row = {"timestamp": rt}
#         for col in non_timestamp:
#             interp_val = np.interp(rt, ts, predictions[col])
#             row[col] = float(interp_val)
#         results.append(row)
#     return results

# --- Modify main interpolation routine to support the new methods ---
def process_interpolation(input_dir, output_dir, selected_columns, explanatory_columns, iteration, method, outliers):
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[selected_columns]
        df['group'] = (df['operation'] != df['operation'].shift()).cumsum()
    
        all_interpolated = []
    
        for group_id, group_df in df.groupby('group'):
            group_df = group_df.reset_index(drop=True)
            start_time = group_df.loc[0, 'timestamp']
            group_df['timestamp'] = group_df['timestamp'] - start_time
            num_points = len(group_df)
            if num_points > 1:
                new_times = np.sort(np.random.uniform(0, group_df['timestamp'].iloc[-1], num_points))
            else:
                new_times = group_df['timestamp'].values
            # Extract interpolation data and remove outliers if requested.
            interp_df = group_df[['timestamp'] + explanatory_columns]
            if outliers == 'moving_average':
                interp_df = remove_outliers(interp_df, method='moving_average')
            elif outliers == 'ransac':
                interp_df = remove_outliers(interp_df, method='ransac')
            elif outliers == "wavelet":
                interp_df = noise_reduction_wavelet(interp_df, explanatory_columns)
            elif outliers == "kalman":
                interp_df = noise_reduction_kalman(interp_df, explanatory_columns)
            elif outliers == "savgol":
                interp_df = noise_reduction_savgol(interp_df, explanatory_columns, window_length=5, polyorder=2)
            elif outliers == 'both':
                interp_df = remove_outliers(interp_df, method='moving_average')
                interp_df = remove_outliers(interp_df, method='ransac')
                # Apply noise reduction methods.
                interp_df = noise_reduction_wavelet(interp_df, explanatory_columns)
                interp_df = noise_reduction_kalman(interp_df, explanatory_columns)
                interp_df = noise_reduction_savgol(interp_df, explanatory_columns, window_length=5, polyorder=2)
            else:
                pass
            # Extended with new methods: 'arima' and 'bayesian'
            if method == 'linear':
                interp_result  = interpolate_results(new_times, interp_df)
            elif method == 'rbf_inverse':
                interp_result  = rbf_adapt(new_times, interp_df, function='inverse', scale_factor=1.0)
            elif method == 'rbf_gaussian':
                interp_result  = rbf_adapt(new_times, interp_df, function='gaussian', scale_factor=1.0)
            elif method == 'rbf_multiquadric':
                interp_result  = rbf_adapt(new_times, interp_df, function='multiquadric', scale_factor=1.0)
            elif method == 'rbf_linear':
                interp_result  = rbf_adapt(new_times, interp_df, function='linear', scale_factor=1.0)
            elif method == 'rbf_cubic':
                interp_result  = rbf_adapt(new_times, interp_df, function='cubic', scale_factor=1.0)
            elif method == 'cubic':
                interp_result  = cubic_implementation(new_times, interp_df)
            elif method == 'bspline':
                interp_result  = bspline_interpolation(new_times, interp_df, degree=5)
            elif method == 'gpr':
                interp_result  = gpr_interpolation(new_times, interp_df)
            elif method == 'curve_fit':
                interp_result  = curve_fit_implementation(new_times, interp_df)
            elif method == 'fft':
                interp_result  = fft_implementation(new_times, interp_df, k=1, window_choice='hann')
            elif method == 'high_performance':
                interp_result  = linear_high_performance(new_times, interp_df)
            elif method == 'precision':
                interp_result  = linear_precision(new_times, interp_df)
            elif method == 'arima':
                interp_result  = arima_interpolation(new_times, interp_df)
            else:
                raise ValueError("Invalid interpolation method")
            
            for row in interp_result:
                row['operation'] = group_df.loc[0, 'operation']
            all_interpolated.extend(interp_result)
    
        result_df = pd.DataFrame(all_interpolated).drop(columns=['timestamp'])
        output_file = os.path.join(output_dir, f"{os.path.basename(csv_file)}_{iteration}.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}")


# --- Main loop ---
def generate_until_condition(input_dir, output_dir, selected_columns, explanatory_columns, interp_method,outliers):
    iteration = 0
    delete_csv_files(output_dir)
    volume_checker = True
    while volume_checker:
        print(f"Iteration: {iteration}")
        process_interpolation(input_dir, output_dir, selected_columns, explanatory_columns, iteration, interp_method,outliers)
        print("Iteration completed. Checking condition...\n")
        iteration += 1
        volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500)
    # Continue removing rows until check_virtual_volume returns True.
    while not check_virtual_volume(rootdir, virtpath, limit_mb=500):
        csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
        if not csv_files:
            break  # No files found, exit the loop.
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Only process if there are at least two rows (keep first row intact)
            if len(df) > 1:
                idx_candidates = df.index[1:].tolist()
                if idx_candidates:
                    # Delete 10 random rows (or fewer if not enough rows available)
                    idx_to_drop = random.sample(idx_candidates, min(50, len(idx_candidates)))
                    df = df.drop(idx_to_drop)
                    df.to_csv(csv_file, index=False)
        print("Randomly deleted 10 rows from each file. Rechecking volume ...")
    F1score = HAR_evaluation(interp_method+"_"+outliers,output_dir)
    print("Condition met. Generation stopped.")
    return F1score

if __name__ == "__main__":
    input_dir = os.path.join(rootdir, 'data/real')
    output_dir = os.path.join(rootdir, 'data/test')
    selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                        'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                        'timestamp','operation']
    explanatory_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                           'atr02/acc_x','atr02/acc_y','atr02/acc_z']
    F1score_list = []
    # Choose from methods: "high_performance", "precision", "fft", "bspline", "gpr", etc.
    methods = ["bspline", "gpr", "curve_fit", ]
    outliers_methods = ["moving_average", "ransac","wavelet", "kalman","savgol","both","none"]
    for m in methods:
        for outliers in outliers_methods:
            print(f"Method: {m}, Outliers: {outliers}")
            F1score = generate_until_condition(input_dir, output_dir, selected_columns, explanatory_columns, m,outliers)
            F1score_list.append(F1score)
    print(F1score_list)