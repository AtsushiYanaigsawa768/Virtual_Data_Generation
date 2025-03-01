import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
input_dir = '/root/Virtual_Data_Generation/data/real'
operation_segment_counts_csv = 'operation_segment_counts.csv'
action_segment_counts_csv = 'action_segment_counts.csv'
combined_segment_counts_csv = 'operation_action_segment_counts.csv'
operation_stats_csv = 'operation_segment_statistics.csv'
action_stats_csv = 'action_segment_statistics.csv'
operation_robust_stats_csv = 'operation_segment_stats_robust.csv'
action_robust_stats_csv = 'action_segment_stats_robust.csv'
operation_plot_png = 'operation_segment_lengths_robust.png'
action_plot_png = 'action_segment_lengths_robust.png'

operation_segments = []
action_segments = []
combined_segments = []

csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
for file in csv_files:
    df = pd.read_csv(file)
    
    if 'actionn' in df.columns and 'action' not in df.columns:
        df = df.rename(columns={'actionn': 'action'})
    
    # Check for required columns
    has_operation = 'operation' in df.columns
    has_action = 'action' in df.columns
    
    if not has_operation and not has_action:
        print(f"Skipping {file}: missing both 'operation' and 'action' columns")
        continue
    
    # Process operation segments
    if has_operation:
        op_change = (df['operation'] != df['operation'].shift(1))
        df['op_group_id'] = op_change.cumsum()
        
        op_group_stats = df.groupby('op_group_id').agg(
            operation=('operation', 'first'),
            count=('op_group_id', 'size')
        ).reset_index(drop=True)
        
        operation_segments.append(op_group_stats)
    
    # Process action segments
    if has_action:
        action_change = (df['action'] != df['action'].shift(1))
        df['action_group_id'] = action_change.cumsum()
        
        action_group_stats = df.groupby('action_group_id').agg(
            action=('action', 'first'),
            count=('action_group_id', 'size')
        ).reset_index(drop=True)
        
        action_segments.append(action_group_stats)
    
    # Process combined operation-action segments
    if has_operation and has_action:
        df['op_action_key'] = df['operation'].astype(str) + '_' + df['action'].astype(str)
        combined_change = (df['op_action_key'] != df['op_action_key'].shift(1))
        df['combined_group_id'] = combined_change.cumsum()
        
        combined_group_stats = df.groupby('combined_group_id').agg(
            operation=('operation', 'first'),
            action=('action', 'first'),
            count=('combined_group_id', 'size')
        ).reset_index(drop=True)
        
        combined_segments.append(combined_group_stats)

# Combine and save segment data
if operation_segments:
    all_operation_segments = pd.concat(operation_segments, ignore_index=True)
    all_operation_segments.to_csv(operation_segment_counts_csv, index=False)
    print(f"Operation segment counts saved to: {operation_segment_counts_csv}")

if action_segments:
    all_action_segments = pd.concat(action_segments, ignore_index=True)
    all_action_segments.to_csv(action_segment_counts_csv, index=False)
    print(f"Action segment counts saved to: {action_segment_counts_csv}")

if combined_segments:
    all_combined_segments = pd.concat(combined_segments, ignore_index=True)
    all_combined_segments.to_csv(combined_segment_counts_csv, index=False)
    print(f"Combined segment counts saved to: {combined_segment_counts_csv}")

# Process operation statistics
if operation_segments:
    # Standard statistics for operations
    op_stats = all_operation_segments.groupby('operation')['count'].agg(
        mean='mean',
        std='std'
    ).reset_index()
    op_stats['std'] = op_stats['std'].fillna(0)
    op_stats.to_csv(operation_stats_csv, index=False)
    print(f"Operation segment statistics saved to: {operation_stats_csv}")
    
    # Robust statistics for operations
    operations = sorted(all_operation_segments['operation'].unique())
    op_robust_stats = {}
    op_robust_segments = []
    
    for op in operations:
        counts = all_operation_segments.loc[all_operation_segments['operation'] == op, 'count']
        Q1 = counts.quantile(0.25)
        Q3 = counts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        counts_no_outliers = counts[(counts >= lower_bound) & (counts <= upper_bound)]
        
        median = counts_no_outliers.median()
        mad = np.median(np.abs(counts_no_outliers - median))
        robust_std = counts_no_outliers.std()
        
        op_robust_stats[op] = {
            'median': median,
            'mad': mad,
            'std': robust_std,
            'n_without_outliers': counts_no_outliers.count()
        }
        
        df_no_out = all_operation_segments[(all_operation_segments['operation'] == op) & 
                                (all_operation_segments['count'] >= lower_bound) &
                                (all_operation_segments['count'] <= upper_bound)]
        op_robust_segments.append(df_no_out)
    
    # Save robust statistics for operations
    op_robust_stats_df = pd.DataFrame([
        {'operation': op,
         'median': info['median'],
         'std': info['std'],
         'n_without_outliers': info['n_without_outliers']}
        for op, info in op_robust_stats.items()
    ])
    op_robust_stats_df.to_csv(operation_robust_stats_csv, index=False)
    print(f"Robust operation segment statistics saved to: {operation_robust_stats_csv}")
    
    # Plot for operations
    plt.figure(figsize=(10, 6))
    for idx, op in enumerate(operations):
        df_no_out = pd.concat([seg for seg in op_robust_segments if seg['operation'].iloc[0] == op])
        counts = df_no_out['count'].values
        y_coords = np.full_like(counts, idx)
        plt.plot(counts, y_coords, 'o', label=op)
    
    plt.yticks(range(len(operations)), operations)
    plt.xlabel('Segment Length (without outliers)')
    plt.title('Operation Segment Lengths (Robust)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(operation_plot_png)
    plt.close()
    print(f"Operation segment lengths robust plot saved to: {operation_plot_png}")

# Process action statistics
if action_segments:
    # Standard statistics for actions
    action_stats = all_action_segments.groupby('action')['count'].agg(
        mean='mean',
        std='std'
    ).reset_index()
    action_stats['std'] = action_stats['std'].fillna(0)
    action_stats.to_csv(action_stats_csv, index=False)
    print(f"Action segment statistics saved to: {action_stats_csv}")
    
    # Robust statistics for actions
    actions = sorted(all_action_segments['action'].unique())
    action_robust_stats = {}
    action_robust_segments = []
    
    for act in actions:
        counts = all_action_segments.loc[all_action_segments['action'] == act, 'count']
        Q1 = counts.quantile(0.25)
        Q3 = counts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        counts_no_outliers = counts[(counts >= lower_bound) & (counts <= upper_bound)]
        
        median = counts_no_outliers.median()
        mad = np.median(np.abs(counts_no_outliers - median))
        robust_std = counts_no_outliers.std()
        
        action_robust_stats[act] = {
            'median': median,
            'mad': mad,
            'std': robust_std,
            'n_without_outliers': counts_no_outliers.count()
        }
        
        df_no_out = all_action_segments[(all_action_segments['action'] == act) & 
                                (all_action_segments['count'] >= lower_bound) &
                                (all_action_segments['count'] <= upper_bound)]
        action_robust_segments.append(df_no_out)
    
    # Save robust statistics for actions
    action_robust_stats_df = pd.DataFrame([
        {'action': act,
         'median': info['median'],
         'std': info['std'],
         'n_without_outliers': info['n_without_outliers']}
        for act, info in action_robust_stats.items()
    ])
    action_robust_stats_df.to_csv(action_robust_stats_csv, index=False)
    print(f"Robust action segment statistics saved to: {action_robust_stats_csv}")
    
    # Plot for actions
    plt.figure(figsize=(10, 6))
    for idx, act in enumerate(actions):
        df_no_out = pd.concat([seg for seg in action_robust_segments if seg['action'].iloc[0] == act])
        counts = df_no_out['count'].values
        y_coords = np.full_like(counts, idx)
        plt.plot(counts, y_coords, 'o', label=act)
    
    plt.yticks(range(len(actions)), actions)
    plt.xlabel('Segment Length (without outliers)')
    plt.title('Action Segment Lengths (Robust)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(action_plot_png)
    plt.close()
    print(f"Action segment lengths robust plot saved to: {action_plot_png}")
