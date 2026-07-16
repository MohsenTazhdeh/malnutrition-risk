from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ArtifactWriter:
    out_dir: Path

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.written: list[Path] = []

    def write_metrics(self, metrics: dict[str, float], name: str = 'metrics.json') -> Path:
        return self.write_json(metrics, name)

    def write_json(self, obj: dict, name: str) -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return self._record_path(path)

    def write_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.out_dir / name
        df.to_parquet(path)
        return self._record_path(path)

    def write_figure(self, fig, name: str, dpi: int = 300) -> Path:
        path = self.out_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return self._record_path(path)

    def write_table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.out_dir / name
        df.to_csv(path, index=False)
        return self._record_path(path)

    def _record_path(self, path: Path) -> Path:
        self.written.append(path)
        return path
