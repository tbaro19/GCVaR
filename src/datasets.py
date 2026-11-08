"""Deterministic local-file operational dataset adapters for GCVaR.

No network downloads. Validate original data SHA externally and save manifest.
Split chronologically before any calibration; no holdout-driven fitting.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

DATASET_SOURCES = {
    'bike': {
        'url': 'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
        'doi': '10.24432/C5W894',
        'expected_rows': 17389
    },
    'tetouan': {
        'url': 'https://archive.ics.uci.edu/dataset/849/power+consumption+',
        'doi': '10.24432/C5B034',
        'expected_rows': 52417
    },
    'electricity': {
        'url': 'https://archive.ics.uci.edu/dataset/321/electricityl',
        'doi': '10.24432/C58C86',
        'expected_customers': 370
    },
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_bike(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'dteday', 'hr', 'casual', 'registered'}
    if not required.issubset(df.columns):
        raise ValueError(f'Bike missing {required - set(df.columns)}')
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values(['timestamp'], kind='stable').reset_index(drop=True)
    for c in ('casual', 'registered'):
        df[c] = pd.to_numeric(df[c], errors='raise')
    if 'cnt' in df and not np.array_equal((df['casual'] + df['registered']).to_numpy(), df['cnt'].to_numpy()):
        raise ValueError('Bike cnt != casual+registered')
    return df[['timestamp', 'casual', 'registered']].rename(columns={'casual': 'factor_1', 'registered': 'factor_2'})


def load_tetouan(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    names = list(df.columns)
    datecol = next((c for c in names if 'date' in c.lower() or 'time' in c.lower()), None)
    zones = [c for c in names if 'zone' in c.lower() and ('consumption' in c.lower() or 'power' in c.lower())]
    if datecol is None or len(zones) < 2:
        raise ValueError('Tetouan expected date and two zone consumption columns')
    try:
        parsed = pd.to_datetime(df[datecol], dayfirst=True, errors='raise')
    except Exception:
        parsed = pd.to_datetime(df[datecol], dayfirst=True, format='mixed', errors='raise')
    frame = pd.DataFrame({
        'timestamp': parsed,
        'factor_1': pd.to_numeric(df[zones[0]], errors='raise'),
        'factor_2': pd.to_numeric(df[zones[1]], errors='raise')
    })
    return frame.sort_values('timestamp', kind='stable').reset_index(drop=True)


def load_electricity(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal=',')
    if df.shape[1] < 21:
        raise ValueError('Electricity load requires >=20 customer series')
    timestamp_col = df.columns[0]
    times = pd.to_datetime(df[timestamp_col], errors='raise')
    customers = sorted(df.columns[1:])
    if len(customers) < 20:
        raise ValueError('need 20 fixed customers')
    left = customers[:10]
    right = customers[10:20]
    loads = df[left + right].apply(pd.to_numeric, errors='raise')
    if loads.isna().any().any():
        raise ValueError('source contains missing readings: declare preprocessing before use')
    out = pd.DataFrame({
        'timestamp': times,
        'factor_1': loads[left].sum(axis=1),
        'factor_2': loads[right].sum(axis=1)
    })
    return out.sort_values('timestamp', kind='stable').reset_index(drop=True)


def load_local(dataset: str, path: str | Path):
    parser = {'bike': load_bike, 'tetouan': load_tetouan, 'electricity': load_electricity}.get(dataset)
    if parser is None:
        raise ValueError(f'unknown dataset: {dataset}')
    path = Path(path)
    df = parser(path)
    if df[['factor_1', 'factor_2']].isna().any().any():
        raise ValueError('missing factors')
    if not np.isfinite(df[['factor_1', 'factor_2']].to_numpy(dtype=float)).all():
        raise ValueError('non-finite factors')
    if (df[['factor_1', 'factor_2']].to_numpy(dtype=float) < 0).any():
        raise ValueError('negative demand factor')
    return df, {
        'dataset': dataset,
        'raw_path': str(path),
        'raw_sha256': file_sha256(path),
        'N': len(df),
        'factor_columns': ['factor_1', 'factor_2'],
        'source': DATASET_SOURCES.get(dataset, {})
    }


def chronological_split(df: pd.DataFrame):
    n = len(df)
    i = int(0.6 * n)
    j = int(0.8 * n)
    return {
        'train': df.iloc[:i].copy(),
        'validation': df.iloc[i:j].copy(),
        'holdout': df.iloc[j:].copy()
    }

