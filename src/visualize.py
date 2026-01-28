import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

# --- KONFIGURATION: INDIVIDUELLE FORMATIERUNG ---
# Hier definierst du das Format für jede Spalte einzeln.
METRIC_FORMATS = {
    # Zeitmessungen (oft klein, daher genauer)
    'time_sec': '.4f',
    
    # Qualitätsmetriken (2-3 Stellen oft ausreichend)
    'accuracy': '.3f',
    'precision': '.3f',
    'recall': '.3f',
    'f1_score': '.3f',
    'auc': '.4f',
    'fpr': '.5f',
    
    # Hardware-Metriken
    'ram_mb': '.1f',
    'vram_system_peak_mb': '.1f',
    'cpu_percent_avg': '.1f',
    'gpu_util_percent_avg': '.1f',
}

DEFAULT_FORMAT = '.3f'

def annotate_bars(ax, df, x_col, y_col, hue_col=None, order=None, hue_order=None):
    """
    Annotates bars with Mean +/- Std Dev.
    Matches patches to data using coordinate logic and provided ordering.
    """
    # 1. Determine Format
    fmt = METRIC_FORMATS.get(y_col, DEFAULT_FORMAT)

    # 2. Calculate Statistics
    group_cols = [x_col]
    if hue_col and hue_col != x_col:
        group_cols.append(hue_col)
    
    # fillna(0) verhindert Fehler bei leeren Gruppen
    stats = df.groupby(group_cols)[y_col].agg(['mean', 'std']).fillna(0)
    
    is_grouped = (hue_col is not None) and (hue_col != x_col)
    n_x = len(order) if order else 0
    
    for i, p in enumerate(ax.patches):
        if pd.isna(p.get_height()) or p.get_height() <= 0:
            continue
            
        # Determine Category based on Patch Index
        if is_grouped:
            if hue_order is None: continue
            hue_idx = i // n_x
            x_idx = i % n_x
            
            if hue_idx >= len(hue_order) or x_idx >= len(order): continue
            h_val = hue_order[hue_idx]
            x_val = order[x_idx]
            
            try:
                mean_val = stats.loc[(x_val, h_val), 'mean']
                std_val = stats.loc[(x_val, h_val), 'std']
            except KeyError: continue
        else:
            x_idx = i % n_x
            if x_idx >= len(order): continue
            x_val = order[x_idx]
            
            try:
                if isinstance(stats.index, pd.MultiIndex):
                    if hue_col == x_col:
                         mean_val = stats.loc[(x_val, x_val), 'mean']
                         std_val = stats.loc[(x_val, x_val), 'std']
                    else:
                         mean_val = stats.loc[x_val, 'mean']
                         std_val = stats.loc[x_val, 'std']
                else:
                    mean_val = stats.loc[x_val, 'mean']
                    std_val = stats.loc[x_val, 'std']
            except KeyError: continue
        
        # Formatierung anwenden
        label = f"{mean_val:{fmt}}\n±{std_val:{fmt}}"
        
        # Text positionieren
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        offset = _y * 0.02 if _y > 0 else 0.01
        
        ax.text(_x, _y + offset, label, ha="center", va="bottom", fontsize=8, fontweight='bold', color='black')

def generate_benchmark_plots(plot_df: pd.DataFrame, title_suffix: str, output_path: str):
    """
    Core plotting engine. Handles the 4x3 grid layout with Error Bars (Std Dev).
    """
    sns.set_theme(style="whitegrid")
    model_order = sorted(plot_df['model'].unique())
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle(f'AI Model Benchmark: {title_suffix}\n(Mean ± Std. Dev.)', fontsize=20, weight='bold')
    
    # --- 1. PERFORMANCE METRICS (TIME) ---
    mask_train = plot_df['task'].astype(str).str.lower().str.strip() == 'training'
    train_df = plot_df[mask_train]
    
    mask_inf = plot_df['task'].astype(str).str.lower().str.strip() == 'inference'
    inf_df = plot_df[mask_inf]
    
    # Debug-Log statt Print
    logger.debug(f"Plot '{title_suffix}': Found {len(train_df)} rows for Training, {len(inf_df)} for Inference.")

    # Plot 1 (Top-Left): Training Time
    ax_train = axes[0, 0]
    if not train_df.empty:
        sns.barplot(
            ax=ax_train, data=train_df, x='model', y='time_sec', 
            hue='model', palette='viridis', order=model_order, 
            errorbar='sd', dodge=False, legend=False
        )
        annotate_bars(ax_train, train_df, 'model', 'time_sec', hue_col='model', order=model_order)
    else:
        ax_train.text(0.5, 0.5, "NO TRAINING DATA", ha='center', va='center', color='red', fontweight='bold')
    ax_train.set_title('Training Time (s) - Lower is better')

    # Plot 2 (Top-Center): Inference Time
    ax_inf = axes[0, 1]
    if not inf_df.empty:
        sns.barplot(
            ax=ax_inf, data=inf_df, x='model', y='time_sec', 
            hue='model', palette='magma', order=model_order, 
            errorbar='sd', dodge=False, legend=False
        )
        annotate_bars(ax_inf, inf_df, 'model', 'time_sec', hue_col='model', order=model_order)
    ax_inf.set_title('Inference Time (s) - Lower is better')

    # --- 2. HARDWARE UTILIZATION ---
    hw_metrics = [
        ('gpu_util_percent_avg', 'GPU Utilization (%)', 0, 2, 'Blues'),
        ('cpu_percent_avg', 'CPU Utilization (%)', 1, 0, 'Oranges'),
        ('vram_system_peak_mb', 'VRAM Usage (MB)', 1, 1, 'Purples'),
        ('ram_mb', 'RAM Usage (MB)', 1, 2, 'Greens')
    ]
    task_order = sorted(plot_df['task'].unique())

    for col, title, r, c, pal in hw_metrics:
        ax = axes[r, c]
        if col in plot_df.columns:
            sns.barplot(
                ax=ax, data=plot_df, x='model', y=col, 
                hue='task', palette=pal, order=model_order, hue_order=task_order,
                errorbar='sd'
            )
            ax.set_title(title)
            annotate_bars(ax, plot_df, 'model', col, hue_col='task', order=model_order, hue_order=task_order)
            ax.legend(title='Task', fontsize='small')

    # --- 3. QUALITY METRICS (ML SCORES) ---
    if not inf_df.empty:
        metrics = [
            ('accuracy', 'Accuracy', 2, 0), 
            ('precision', 'Precision', 2, 1), 
            ('recall', 'Recall', 2, 2), 
            ('f1_score', 'F1-Score', 3, 0), 
            ('fpr', 'False Positive Rate', 3, 1), 
            ('auc', 'AUC', 3, 2)
        ]
        
        for col, label, r, c in metrics:
            ax = axes[r, c]
            if col in inf_df.columns:
                sns.barplot(
                    ax=ax, data=inf_df, x='model', y=col, 
                    hue='model', palette='coolwarm', order=model_order,
                    errorbar='sd', dodge=False, legend=False
                )
                ax.set_title(f'{label}')
                if col != 'fpr': ax.set_ylim(0, 1.15)
                annotate_bars(ax, inf_df, 'model', col, hue_col='model', order=model_order)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}")
    plt.close()

def generate_plots(df_raw: pd.DataFrame, output_dir: str):
    if df_raw.empty:
        logger.warning("No data provided to generate_plots. Skipping.")
        return

    # Namen normalisieren
    model_map = {'CNN': 'CNN', 'XGB': 'XGBoost', 'LR': 'Logistic Regression', 'SVM_TFIDF': 'Linear SVC'}
    df_raw['model'] = df_raw['model'].map(lambda x: model_map.get(x, x))

    # Datasets unterscheiden
    def distinguish_dataset(row):
        ds_name = row.get('dataset_name', row.get('dataset', ''))
        if "PhiUSIIL" in str(ds_name): return f"{row['model']} PhiUSIIL"
        return row['model']
    df_raw['model'] = df_raw.apply(distinguish_dataset, axis=1)

    # Typ-Konvertierung
    cols_to_convert = ['time_sec', 'ram_mb', 'vram_system_peak_mb', 'cpu_percent_avg', 
                       'gpu_util_percent_avg', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'fpr']
    for col in cols_to_convert:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    logger.info("--- Starting Visualization Generation ---")
    
    # Basismodelle Plot
    base_model_names = ['CNN', 'Logistic Regression', 'Linear SVC', 'XGBoost']
    df_base = df_raw[df_raw['model'].isin(base_model_names)].copy()
    
    if not df_base.empty:
        generate_benchmark_plots(df_base, "Basismodelle", os.path.join(output_dir, "benchmark_basismodelle.png"))
    else:
        logger.warning("No base model data found for overview plot.")

    # Vergleichsplots für Varianten (PhiUSIIL vs. Normal)
    targets = ['CNN', 'Logistic Regression', 'Linear SVC', 'XGBoost']
    for target in targets:
        target_phi = f"{target} PhiUSIIL"
        df_compare = df_raw[df_raw['model'].isin([target, target_phi])].copy()
        
        if not df_compare.empty and len(df_compare['model'].unique()) > 1:
            generate_benchmark_plots(df_compare, f"Compare {target}", os.path.join(output_dir, f"benchmark_compare_{target.replace(' ', '_')}.png"))