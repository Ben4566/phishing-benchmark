import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
import os
from typing import List, Optional, Dict

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

class BenchmarkVisualizer:
    """
    Visualization engine for benchmark analytics.
    
    Responsibilities:
    1. Renders comparative performance metrics (Time, Hardware, Quality).
    2. Overlays statistical significance (Mean ± Std Dev) on visual elements.
    3. Manages output persistence for reporting artifacts.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration: Precision formatting for specific metric types
        self.metric_formats = {
            'time_sec': '.4f', 'accuracy': '.3f', 'precision': '.3f',
            'recall': '.3f', 'f1_score': '.3f', 'auc': '.4f',
            'ram_mb': '.1f', 'gpu_util_percent_avg': '.1f'
        }
        sns.set_theme(style="whitegrid")

    def _annotate_bars(self, ax, df, x_col, y_col, hue_col=None, order=None):
        """
        Overlays value annotations (Mean ± Std) onto Seaborn bar patches.
        
        Complexity Note:
        Seaborn renders patches in a specific order (grouped by Hue, then X). 
        This method maps the linear patch index back to the source DataFrame's 
        (X, Hue) coordinates to retrieve the correct statistical values.
        """
        fmt = self.metric_formats.get(y_col, '.3f')
        
        # 1. Establish ordering to align DataFrame aggregation with Plot patches
        if order is None:
            x_order = sorted(df[x_col].unique())
        else:
            x_order = order
            
        hue_order = None
        if hue_col and hue_col != x_col:
            try:
                hue_order = sorted(df[hue_col].unique())
            except:
                hue_order = df[hue_col].unique()

        # 2. Aggregation: Calculate Mean and Std Dev for annotation
        group_cols = [x_col]
        if hue_order is not None: 
            group_cols.append(hue_col)
        
        try:
            # fillna(0) handles single-run cases where std dev is undefined
            stats = df.groupby(group_cols)[y_col].agg(['mean', 'std']).fillna(0)
        except TypeError:
            return # Abort if column is non-numeric

        num_x = len(x_order)
        
        # 3. Patch Iteration & Text Placement
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            
            # Skip rendering for null or zero-height bars
            if pd.isna(height) or height <= 0: 
                continue
            
            # --- Coordinate Mapping: Patch Index -> (x_val, hue_val) ---
            val_std = 0.0
            
            if hue_order is not None:
                # Resolve 2D mapping (Hue + X)
                hue_idx = i // num_x
                x_idx = i % num_x
                
                if hue_idx < len(hue_order) and x_idx < len(x_order):
                    curr_x = x_order[x_idx]
                    curr_hue = hue_order[hue_idx]
                    try:
                        val_std = stats.loc[(curr_x, curr_hue), 'std']
                    except KeyError:
                        pass
            else:
                # Resolve 1D mapping (X only)
                x_idx = i % num_x
                if x_idx < len(x_order):
                    curr_x = x_order[x_idx]
                    try:
                        val_std = stats.loc[curr_x, 'std']
                    except KeyError:
                        pass
            
            # Layout Calculation
            x_pos = p.get_x() + p.get_width() / 2
            y_pos = 0 + (ax.get_ylim()[1] * 0.02) # Default: Anchor text inside base of bar
            
            # Styling
            color = 'white'
            path_effect = [pe.withStroke(linewidth=2, foreground="black")]
            
            # Contrast Logic: If bar is too short, move text above bar in black
            if height < (ax.get_ylim()[1] * 0.15):
                y_pos = height + (ax.get_ylim()[1] * 0.01)
                color = 'black'
                path_effect = []

            # Label Construction
            if val_std > 0:
                label = f"{height:{fmt}}\n±{val_std:{fmt}}"
            else:
                label = f"{height:{fmt}}"
            
            ax.text(x_pos, y_pos, label, ha="center", va="bottom", 
                    fontsize=7, fontweight='bold', color=color, path_effects=path_effect)

    def plot_overview(self, df: pd.DataFrame, title_suffix: str, filename: str):
        """
        Orchestrates the generation of the comprehensive 4x3 metrics grid.
        Sections: Timing, Hardware Utilization, Quality Metrics.
        """
        if df.empty:
            logger.warning(f"Aborting plot generation: No data found for {filename}")
            return

        fig, axes = plt.subplots(4, 3, figsize=(20, 24))
        fig.suptitle(f'Benchmark Results: {title_suffix}', fontsize=20, weight='bold')
        
        model_order = sorted(df['model'].unique())
        
        # 1. Execution Timing (Train vs Inference)
        self._plot_metric(axes[0, 0], df[df['task'] == 'Training'], 'time_sec', 'Training Time (s)', model_order)
        self._plot_metric(axes[0, 1], df[df['task'] == 'Inference'], 'time_sec', 'Inference Time (s)', model_order)
        
        # 2. Hardware Resource Utilization
        hw_metrics = [
            (0, 2, 'gpu_util_percent_avg', 'GPU Util (%)'),
            (1, 0, 'cpu_percent_avg', 'CPU Util (%)'),
            (1, 1, 'vram_mb', 'VRAM (MB)'),
            (1, 2, 'ram_mb', 'RAM Usage (MB)')
        ]
        for r, c, col, title in hw_metrics:
            self._plot_metric(axes[r, c], df, col, title, model_order, hue='task')

        # 3. Model Quality (Inference Scope Only)
        inf_df = df[df['task'] == 'Inference']
        qual_metrics = [
            (2, 0, 'accuracy', 'Accuracy'), (2, 1, 'precision', 'Precision'),
            (2, 2, 'recall', 'Recall'), (3, 0, 'f1_score', 'F1-Score'),
            (3, 1, 'fpr', 'False Positive Rate'), (3, 2, 'auc', 'AUC')
        ]
        for r, c, col, title in qual_metrics:
            self._plot_metric(axes[r, c], inf_df, col, title, model_order)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visual artifact saved: {save_path}")
        plt.close()

    def _plot_metric(self, ax, df, y_col, title, order, hue='model'):
        """Helper to render a single subplot with error bars and custom annotations."""
        if df.empty or y_col not in df.columns:
            ax.set_visible(False)
            return
            
        # 'errorbar=sd' instructs Seaborn to calculate and render standard deviation lines
        sns.barplot(
            ax=ax, data=df, x='model', y=y_col, 
            hue=hue, order=order, errorbar='sd', palette='viridis'
        )
        ax.set_title(title)
        ax.set_xlabel("")
        
        if hue == 'model':
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        
        # Apply statistical text overlay
        self._annotate_bars(ax, df, 'model', y_col, hue, order)