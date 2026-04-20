"""Merge Q and P series into final raw/ files for the LSTM pipeline.

Reads station_mapping.yaml and for each station:
  - Loads Q from vazao_series/ or cota_series/ (based on flow_type)
  - Loads P from precipitacao_series/
  - Outer-joins on date, reindexes to full daily range
  - Saves to data/raw/{id}.csv with columns (Date, Q, P)
"""
from pathlib import Path
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA      = ROOT / "data"
SRC       = DATA / "raw_source_ana"
OUT_DIR   = DATA / "raw"
YAML_PATH = ROOT / "config/data_config.yaml"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_series(path: Path, value_col: str, rename_to: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Data"])
    df = df.rename(columns={"Data": "Date", value_col: rename_to})
    df = df[["Date", rename_to]].dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.normalize()
    return df.set_index("Date")


def _merge_station(q_path: Path, p_path: Path, q_col: str) -> pd.DataFrame:
    q = _load_series(q_path, q_col, "Q")
    p = _load_series(p_path, "Precipitacao", "P")
    merged   = q.join(p, how="outer")
    full_idx = pd.date_range(merged.index.min(), merged.index.max(), freq="D")
    merged   = merged.reindex(full_idx)
    merged.index.name = "Date"
    return merged.reset_index()


def _nan_pct(series: pd.Series) -> float:
    return 100.0 * series.isna().sum() / max(len(series), 1)


def _log_station(station_id: int, df: pd.DataFrame) -> dict:
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    q_nan    = _nan_pct(df["Q"])
    p_nan    = _nan_pct(df["P"])
    print(
        f"[{station_id}]  {date_min} → {date_max}  |  "
        f"rows={len(df):,}  |  Q NaN={q_nan:.1f}%  P NaN={p_nan:.1f}%"
    )
    return {
        "id": station_id, "start": str(date_min), "end": str(date_max),
        "rows": len(df), "q_nan_%": round(q_nan, 1), "p_nan_%": round(p_nan, 1),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_raw_dataset(yaml_path: Path = YAML_PATH) -> list[dict]:
    """Gera os arquivos finais em data/raw/ para todas as estações do mapeamento.

    Lê station_mapping.yaml, faz o merge Q+P de cada estação e salva
    em data/raw/{id}.csv. Retorna o resumo consolidado como lista de dicts.

    Args:
        yaml_path: caminho para o station_mapping.yaml (usa o padrão do
                   projeto se omitido).

    Returns:
        Lista de dicts com id, start, end, rows, q_nan_%, p_nan_% por estação.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stations = cfg.get("stations", [])
    if not stations:
        print("Nenhuma estação encontrada em station_mapping.yaml.")
        return []

    summary = []

    for stn in stations:
        station_id = stn["id"]
        flow_type  = stn["flow_type"]
        precip_id  = stn["precip_id"]

        q_dir  = SRC / f"{flow_type}_series"
        p_dir  = SRC / "precipitacao_series"
        q_col  = "Vazao" if flow_type == "vazao" else "Cota"
        q_path = q_dir / f"{station_id}.csv"
        p_path = p_dir / f"{precip_id}.csv"

        missing = [p for p in (q_path, p_path) if not p.exists()]
        if missing:
            print(f"[{station_id}] SKIP — não encontrado: {[str(m) for m in missing]}")
            continue

        df = _merge_station(q_path, p_path, q_col)
        df.to_csv(OUT_DIR / f"{station_id}.csv", index=False, date_format="%Y-%m-%d")
        summary.append(_log_station(station_id, df))

    if summary:
        print("\n" + "═" * 70)
        print(f"{'RESUMO':^70}")
        print("═" * 70)
        print(f"{'Estação':<12} {'Início':<12} {'Fim':<12} {'Dias':>7}  {'Q NaN%':>8}  {'P NaN%':>8}")
        print("─" * 70)
        for r in summary:
            print(f"{r['id']:<12} {r['start']:<12} {r['end']:<12} "
                  f"{r['rows']:>7,}  {r['q_nan_%']:>7.1f}%  {r['p_nan_%']:>7.1f}%")
        print("═" * 70)
        print(f"Total processado: {len(summary)}/{len(stations)} estações\n")

    return summary


if __name__ == "__main__":
    build_raw_dataset()