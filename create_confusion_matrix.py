import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from evaluation import HAR_evaluation, data_loader_OpenPack, Transformer, vote_labels

def create_and_save_confusion_matrix(model_name, virt_directory="/root/Virtual_Data_Generation/data/virtual"):
    """
    指定されたモデルで予測を行い、混同行列を作成して保存します。
    
    Parameters:
    -----------
    model_name : str
        モデルの名前（保存ファイル名に使用されます）
    virt_directory : str
        仮想データのディレクトリパス
    """
    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # HAR_evaluationを実行してF1スコアを取得
    # この関数を少し修正して、必要なデータを返すようにすることも可能
    f1 = HAR_evaluation(model_name, virt_directory)
    
    # Resultsディレクトリを確認
    results_dir = '/root/Virtual_Data_Generation/Results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存されたprediction_resultsがあれば読み込む
    results_file = f'{results_dir}/{model_name}_prediction_results.csv'
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        
        # 混同行列を作成
        true_labels = results_df['True Label'].values
        pred_labels = results_df['Predicted Label'].values
        
        # ユニークなラベルを取得
        labels = sorted(list(set(list(true_labels) + list(pred_labels))))
        
        # 混同行列の計算
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        
        # 混同行列の可視化と保存
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name} (F1: {f1:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_file = f'{results_dir}/{model_name}_confusion_matrix_detailed.png'
        plt.savefig(cm_file)
        print(f"保存しました: {cm_file}")
        
        # 分類レポートも出力
        report = classification_report(true_labels, pred_labels, labels=labels)
        print("\n分類レポート:")
        print(report)
        
        # 分類レポートをファイルに保存
        with open(f'{results_dir}/{model_name}_classification_report.txt', 'w') as f:
            f.write(report)
    else:
        print(f"予測結果ファイルが見つかりません: {results_file}")
        print("まずHAR_evaluation関数を実行して予測結果を保存してください。")

if __name__ == "__main__":
    # 使用例
    model_name = input("評価するモデルの名前を入力してください: ")
    create_and_save_confusion_matrix(model_name)
