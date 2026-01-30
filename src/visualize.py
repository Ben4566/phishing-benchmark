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
    Klasse für die Erstellung von Benchmark-Plots.
    Diese Klasse ist 'generisch' und kennt keine spezifischen Modellnamen.
    Sie plottet einfach das, was im DataFrame steht.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Standard-Formate
        self.metric_formats = {
            'time_sec': '.4f', 'accuracy': '.3f', 'precision': '.3f',
            'recall': '.3f', 'f1_score': '.3f', 'auc': '.4f',
            'ram_mb': '.1f', 'gpu_util_percent_avg': '.1f'
        }
        sns.set_theme(style="whitegrid")

    def _annotate_bars(self, ax, df, x_col, y_col, hue_col=None, order=None):
        """Hilfsmethode für Text-Annotationen auf Balken."""
        fmt = self.metric_formats.get(y_col, '.3f')
        
        # Berechnung von Mean/Std für Annotationen
        group_cols = [x_col]
        if hue_col and hue_col != x_col: group_cols.append(hue_col)
        
        # Aggregation
        try:
            stats = df.groupby(group_cols)[y_col].agg(['mean', 'std']).fillna(0)
        except TypeError:
            return # Fallback falls Daten nicht numerisch

        is_grouped = (hue_col is not None) and (hue_col != x_col)
        
        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height) or height <= 0: continue
            
            # Positionierung
            x_pos = p.get_x() + p.get_width() / 2
            y_pos = 0 + (ax.get_ylim()[1] * 0.02) # Standard: Unten
            
            # Text Farbe & Style
            color = 'white'
            path_effect = [pe.withStroke(linewidth=2, foreground="black")]
            
            # Wenn Balken zu klein, Text darüber schreiben
            if height < (ax.get_ylim()[1] * 0.15):
                y_pos = height + (ax.get_ylim()[1] * 0.01)
                color = 'black'
                path_effect = []

            # Wert ermitteln (Trickreich bei Seaborn Barplots)
            label = f"{height:{fmt}}" 
            
            ax.text(x_pos, y_pos, label, ha="center", va="bottom", 
                    fontsize=8, fontweight='bold', color=color, path_effects=path_effect)

    def plot_overview(self, df: pd.DataFrame, title_suffix: str, filename: str):
        """Erstellt das 4x3 Grid Layout."""
        if df.empty:
            logger.warning(f"Keine Daten für Plot: {filename}")
            return

        fig, axes = plt.subplots(4, 3, figsize=(20, 24))
        fig.suptitle(f'Benchmark Results: {title_suffix}', fontsize=20, weight='bold')
        
        model_order = sorted(df['model'].unique())
        
        # 1. Performance (Time)
        self._plot_metric(axes[0, 0], df[df['task'] == 'Training'], 'time_sec', 'Training Time (s)', model_order)
        self._plot_metric(axes[0, 1], df[df['task'] == 'Inference'], 'time_sec', 'Inference Time (s)', model_order)
        
        # 2. Hardware
        hw_metrics = [
            (0, 2, 'gpu_util_percent_avg', 'GPU Util (%)'),
            (1, 0, 'cpu_percent_avg', 'CPU Util (%)'),
            (1, 1, 'vram_system_peak_mb', 'VRAM Peak (MB)'),
            (1, 2, 'ram_mb', 'RAM Usage (MB)')
        ]
        for r, c, col, title in hw_metrics:
            self._plot_metric(axes[r, c], df, col, title, model_order, hue='task')

        # 3. Quality (Nur Inference)
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
        logger.info(f"Plot gespeichert: {save_path}")
        plt.close()

    def _plot_metric(self, ax, df, y_col, title, order, hue='model'):
        if df.empty or y_col not in df.columns:
            ax.set_visible(False)
            return
            
        sns.barplot(
            ax=ax, data=df, x='model', y=y_col, 
            hue=hue, order=order, errorbar='sd', palette='viridis'
        )
        ax.set_title(title)
        ax.set_xlabel("")
        
        # Sicherheits-Check: Nur löschen, wenn Legende existiert
        if hue == 'model':
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        
        self._annotate_bars(ax, df, 'model', y_col, hue, order)