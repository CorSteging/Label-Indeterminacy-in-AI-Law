import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from collections import defaultdict

method_key = {
    'wcorr': 'corr',
    'obs': 'obs',
    'obs_ip': 'obs+ip',
    'nn': 'nn',
    'mexpall': 'exp_all',
    'mexpavg': 'exp_avg',
    'mexpmax': 'exp_max',
    'mexpmin': 'exp_min',
    'mexpagr': 'exp_agr',
}

def find_method_files(results_dir):
    result_files = [f for f in os.listdir(results_dir) if f.startswith("results_split_") and f.endswith(".json")]

    method_files = defaultdict(list)
    pattern = re.compile(r"results_split_(\d+)_method_(.+?)\.json")
    
    for fname in result_files:
        match = pattern.match(fname)
        if match:
            split_id = match.group(1)
            method_name = match.group(2)
            method_files[method_name].append(fname)
    return method_files

def append_avg_predictions_to_datasets(all_datasets, results_dir):
    """
    For each test dataset and method, load prediction files across splits,
    compute the average and standard deviation of predictions,
    and append them as new columns to the datasets.

    Parameters:
    - all_datasets: dict with test datasets as pandas DataFrames
    - results_dir: directory path containing the JSON result files

    Returns:
    - updated all_datasets with appended average and std prediction columns
    """
    method_files = find_method_files(results_dir)

    for test_set_name, df in all_datasets['test'].items():
        print(f"\n📊 Processing test set: {test_set_name}")

        for method, files in method_files.items():
            print(f"  🔍 Averaging predictions for method: {method}")
            preds_across_splits = []

            for file in sorted(files):
                try:
                    with open(results_dir + file, 'r') as f:
                        data = json.load(f)

                    key = f"{test_set_name}_test_predictions"
                    if key not in data or "predictions" not in data[key]:
                        print(f"    ⚠️  Warning: missing '{key}' or inner keys in {file} — skipping.")
                        continue

                    preds = data[key]["predictions"]
                    preds_across_splits.append(np.array(preds))

                except Exception as e:
                    print(f"    ❌ Failed to process {file}: {e}")

            if preds_across_splits:
                stacked_preds = np.stack(preds_across_splits)
                avg_preds = stacked_preds.mean(axis=0)
                std_preds = stacked_preds.std(axis=0)

                # Add average and std predictions as columns
                if avg_preds.ndim == 1 or avg_preds.shape[1] == 1:
                    df[f'avg_pred_{method}'] = avg_preds[:, 0] if avg_preds.ndim > 1 else avg_preds
                    df[f'std_pred_{method}'] = std_preds[:, 0] if std_preds.ndim > 1 else std_preds
                else:
                    for i in range(avg_preds.shape[1]):
                        df[f'avg_pred_{method}_class_{i}'] = avg_preds[:, i]
                        df[f'std_pred_{method}_class_{i}'] = std_preds[:, i]
            else:
                print(f"    ⚠️  Warning: No predictions found for method {method}, test set {test_set_name}")

    return all_datasets

def load_all_results(results_folder='results'):
    files = glob(f"{results_folder}/*.json")
    all_records = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            record = {
                'split_id': data['split_id'],
                'method': data['method'],
                'chamber_test_metrics': data['chamber_test_metrics'],
                'grand_chamber_test_metrics': data['grand_chamber_test_metrics'],
                'chamber_test_predictions': data['chamber_test_predictions'],      # list of preds
                'grand_chamber_test_predictions': data['grand_chamber_test_predictions']  # list of preds
            }
            all_records.append(record)
    return pd.DataFrame(all_records)



def plot_prediction_distributions(all_datasets, methods, test_set_name, filename='analysis/violin_plots.png', palette="Set2"):
    """
    Generate violin plots of predictions per method with 4 plots on top row and 5 on bottom row.
    Uses global `method_key` for titles.
    """
    assert len(methods) == 9, "This layout expects exactly 9 methods."
    sns.set(style="whitegrid")

    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], hspace=0.15, wspace=0.15)

    axes = []
    for col in range(4):
        axes.append(fig.add_subplot(gs[0, col]))
    for col in range(5):
        axes.append(fig.add_subplot(gs[1, col]))

    if isinstance(palette, str):
        color_list = sns.color_palette(palette, 9)
    else:
        color_list = palette

    for i, method in enumerate(methods):
        ax = axes[i]
        pred_col = f'avg_pred_{method}'
        dfs_to_plot = []

        if test_set_name == 'all':
            chamber_df = all_datasets['test'].get('chamber')
            grand_df = all_datasets['test'].get('grand_chamber')

            frames = []
            if chamber_df is not None and pred_col in chamber_df:
                frames.append(chamber_df[[pred_col]].copy())
            if grand_df is not None and pred_col in grand_df:
                frames.append(grand_df[[pred_col]].copy())

            if not frames:
                print(f"⚠️  No predictions found for method '{method}' in combined test sets.")
                continue

            combined_df = pd.concat(frames, ignore_index=True)
            combined_df.rename(columns={pred_col: "Prediction"}, inplace=True)

            sns.violinplot(
                y="Prediction",
                data=combined_df,
                ax=ax,
                inner=None,
                color=color_list[i],
                linewidth=0,
                legend=False
            )
            for violin_body in ax.collections:
                violin_body.set_alpha(0.5)

            vals = combined_df['Prediction'].values
            ymin, ymax = vals.min(), vals.max()
            ymean = vals.mean()
            
            x_center = 0  
            
            ax.hlines([ymin, ymean, ymax], x_center - 0.2, x_center + 0.2, colors='black', linestyles='solid', linewidth=1.5)
            
            ax.vlines(x_center, ymin, ymax, colors='black', linestyles='solid', linewidth=1.2)

            ax.set_title(method_key.get(method, method), fontsize=20, pad=10)
            ax.set_xlabel("")
            if i == 0 or i == 4:
                ax.set_ylabel("Prediction", fontsize=20)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, 0.5)  

        else:
            if test_set_name in ['chamber', 'both']:
                chamber_df = all_datasets['test'].get('chamber')
                if chamber_df is not None and pred_col in chamber_df:
                    temp = chamber_df[[pred_col]].copy()
                    temp['TestSet'] = 'Chamber'
                    dfs_to_plot.append(temp)

            if test_set_name in ['grand_chamber', 'both']:
                grand_df = all_datasets['test'].get('grand_chamber')
                if grand_df is not None and pred_col in grand_df:
                    temp = grand_df[[pred_col]].copy()
                    temp['TestSet'] = 'Grand Chamber'
                    dfs_to_plot.append(temp)

            if not dfs_to_plot:
                print(f"⚠️  No predictions found for method '{method}' in selected test set(s).")
                continue

            combined_df = pd.concat(dfs_to_plot, ignore_index=True)
            combined_df.rename(columns={pred_col: "Prediction"}, inplace=True)

            sns.violinplot(
                x="TestSet",
                y="Prediction",
                hue="TestSet",
                data=combined_df,
                ax=ax,
                inner="quartile",
                palette=palette,
                legend=False
            )
            for violin_body in ax.collections:
                violin_body.set_alpha(0.5)

            groups = combined_df.groupby("TestSet")["Prediction"]
            xpos_map = {'Chamber': 0, 'Grand Chamber': 1}
            hl_length = 0.1

            for group_name, vals in groups:
                vmin, vmax, vmean = vals.min(), vals.max(), vals.mean()
                xcenter = xpos_map.get(group_name, 0)

                ax.axvline(x=xcenter, color='k', linestyle='-', linewidth=1)
                ax.hlines([vmin, vmean, vmax],
                          xmin=xcenter - hl_length, xmax=xcenter + hl_length,
                          colors='k',
                          linestyles=['--', '-', '--'],
                          linewidth=1)

            ax.set_title(method_key.get(method, method), fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Prediction", fontsize=10)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def add_baseline_rows(df: pd.DataFrame, body: str = 'grand_chamber') -> pd.DataFrame:
    """
    Adds a baseline row for each unique split_id to the DataFrame.
    The baseline predicts the majority class for each split.
    
    Returns a new DataFrame with baseline rows added.
    """
    import copy
    new_rows = []
    grouped = df.groupby('split_id')

    for split_id, group in grouped:
        all_labels = []
        for preds in group[f'{body}_test_predictions']:
            all_labels.extend(preds['labels'])
        
        if not all_labels:
            continue
        
        majority_class = int(sum(all_labels) >= len(all_labels) / 2)

        for _, row in group.iterrows():
            labels = row[f'{body}_test_predictions']['labels']
            preds = [majority_class] * len(labels)
            baseline_row = {
                'method': 'baseline',
                'split_id': split_id,
                f'{body}_test_predictions': {
                    'predictions': preds,
                    'labels': labels
                }
            }
            new_rows.append(baseline_row)
            break  
            
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def evaluate_metrics_per_method(
    df: pd.DataFrame,
    methods: list[str] = None,
    body: str = 'grand_chamber',
    filename: str = None
) -> pd.DataFrame:
    """
    Compute normalized accuracy, F1-score, and MCC (mean ± std) per method.
    
    Args:
        df: DataFrame with:
            - 'method': str
            - 'grand_chamber_test_predictions': dict with keys 'predictions' and 'labels'
        methods: Optional list of method names to filter and order results.
        filename: Optional path to save resulting CSV.

    Returns:
        DataFrame with index=method and columns:
            ['accuracy', 'f1', 'mcc'] with values like "93.21 ± 4.12"
    """


    df = add_baseline_rows(df, body=body)    
    results = defaultdict(lambda: {'accuracy': [], 'f1': [], 'mcc': []})
    
    for _, row in df.iterrows():
        method = row['method']
        preds = row[body+'_test_predictions']['predictions']
        labels = row[body+'_test_predictions']['labels']
        
        if not preds or not labels:
            continue
        
        preds_binary = [1 if p >= 0.5 else 0 for p in preds]
        
        acc = accuracy_score(labels, preds_binary)
        f1 = f1_score(labels, preds_binary)
        mcc = matthews_corrcoef(labels, preds_binary)
        
        results[method]['accuracy'].append(acc)
        results[method]['f1'].append(f1)
        results[method]['mcc'].append(mcc)
    
    summary = []
    for method, scores in results.items():
        acc_mean, acc_std = np.mean(scores['accuracy']), np.std(scores['accuracy'])
        f1_mean, f1_std = np.mean(scores['f1']), np.std(scores['f1'])
        mcc_mean, mcc_std = np.mean(scores['mcc']), np.std(scores['mcc'])
        
        summary.append({
            'method': method,
            'accuracy': f"{round(acc_mean*100, 2)} ± {round(acc_std*100, 2)}",
            'f1':       f"{round(f1_mean*100, 2)} ± {round(f1_std*100, 2)}",
            'mcc':      f"{round(mcc_mean*100, 2)} ± {round(mcc_std*100, 2)}",
        })
    
    df_summary = pd.DataFrame(summary).set_index('method')
    
    if methods is not None:
        df_summary = df_summary.loc[[m for m in methods if m in df_summary.index]]
    else:
        df_summary = df_summary.sort_index()
    
    if filename:
        df_summary.T.to_csv(filename)

    return df_summary.T

def find_max_diff_instances(df: pd.DataFrame, methods: list, top_n: int = 10) -> pd.DataFrame:
    avg_pred_cols = [f'avg_pred_{m}' for m in methods]
    missing_cols = [col for col in avg_pred_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in DataFrame: {missing_cols}")
    df['max_pred_diff'] = df[avg_pred_cols].max(axis=1) - df[avg_pred_cols].min(axis=1)
    result = df.sort_values(by='max_pred_diff', ascending=False)
    return result.head(top_n).copy()

def plot_method_predictions(instance: pd.Series, methods: list):
    avg_pred_cols = [f'avg_pred_{m}' for m in methods]
    preds = instance[avg_pred_cols]
    palette = sns.color_palette("Set2", len(methods))
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, preds, color=palette, alpha=0.7, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 1)
    plt.ylabel('Average Prediction')
    plt.title(f'Prediction per Method for case {instance["id"]}')
    plt.xticks(rotation=45, ha='right')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', visible=False)
    
    for bar, pred in zip(bars, preds):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{pred:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('analysis/individual_'+instance['id']+'.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()

