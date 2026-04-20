"""Convert raw ANA CSVs into clean daily series.
Reads from data/raw_source_ana/{vazao,cota,precipitacao}/ and writes one
CSV per station into data/raw_source_ana/{...}_series/ with columns
(Data, Vazao|Cota|Precipitacao). Wide monthly rows are melted to daily
rows; for each date the NivelConsistencia=2 value wins over NC=1.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_source_ana"


def _read_ana_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise IOError(f"could not decode {path}")


def _reindex_daily(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.sort_values("Data").reset_index(drop=True)
    if df.empty:
        return df
    full = pd.date_range(df["Data"].min(), df["Data"].max(), freq="D")
    return (
        df.set_index("Data")[[value_col]]
        .reindex(full)
        .rename_axis("Data")
        .reset_index()
    )


def _wide_to_series(df: pd.DataFrame, prefix: str, value_name: str) -> pd.DataFrame:
    df = df[df["MediaDiaria"] == 1].copy()
    df["DataHora"] = pd.to_datetime(df["DataHora"], errors="coerce")
    df = df.dropna(subset=["DataHora"])
    day_cols = [
        f"{prefix}{i:02d}"
        for i in range(1, 32)
        if f"{prefix}{i:02d}" in df.columns
    ]
    melted = df.melt(
        id_vars=["NivelConsistencia", "DataHora"],
        value_vars=day_cols,
        var_name="col",
        value_name=value_name,
    )
    melted["day"] = melted["col"].str[-2:].astype(int)
    melted = melted[melted["day"] <= melted["DataHora"].dt.days_in_month]
    melted["Data"] = melted["DataHora"] + pd.to_timedelta(melted["day"] - 1, unit="D")
    melted[value_name] = pd.to_numeric(
        melted[value_name].astype(str).str.replace(",", "."), errors="coerce"
    )
    melted = melted.sort_values(["Data", "NivelConsistencia"])
    melted = melted.drop_duplicates("Data", keep="last")
    return _reindex_daily(melted[["Data", value_name]], value_name)


def _prcp_to_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"])
    df["Chuva_mm"] = pd.to_numeric(
        df["Chuva_mm"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df = df.sort_values(["Data", "NivelConsistencia"])
    df = df.drop_duplicates("Data", keep="last")
    out = df[["Data", "Chuva_mm"]].rename(columns={"Chuva_mm": "Precipitacao"})
    return _reindex_daily(out, "Precipitacao")


def process_vazao(path: Path) -> pd.DataFrame:
    return _wide_to_series(_read_ana_csv(path), "Vazao", "Vazao")


def process_cota(path: Path) -> pd.DataFrame:
    return _wide_to_series(_read_ana_csv(path), "Cota", "Cota")


def process_precipitacao(path: Path) -> pd.DataFrame:
    return _prcp_to_series(_read_ana_csv(path))


def _run(subdir: str, processor) -> None:
    in_dir  = ROOT / subdir
    out_dir = ROOT / f"{subdir}_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.csv")):
        series = processor(path)
        series.to_csv(out_dir / path.name, index=False, date_format="%Y-%m-%d")
        print(f"{subdir}/{path.name}: {len(series)} rows")


def convert_all_ana_series() -> None:
    """Parse todos os CSVs brutos da ANA e grava as séries diárias limpas.

    Processa as três variáveis (vazão, cota, precipitação) e salva os
    resultados em raw_source_ana/{variavel}_series/.
    """
    _run("vazao",        process_vazao)
    _run("cota",         process_cota)
    _run("precipitacao", process_precipitacao)


if __name__ == "__main__":
    convert_all_ana_series()