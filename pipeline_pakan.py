"""Pra-pemrosesan, model XGBoost, dan sistem fuzzy Mamdani (sama dengan Google Colab)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE = Path(__file__).parent
FITUR_LAMA = ['DOC', 'MBW', 'Size', 'Biomassa', 'Luas', 'pH_Pagi', 'pH_Siang', 'Suhu_Pagi', 'Suhu_Siang']


def _baca_fitur():
    """Daftar fitur dibaca dari metadata model, sehingga model dengan/tanpa 'Pakan' sama-sama didukung."""
    try:
        with open(BASE / 'model_xgb' / 'metadata.json', encoding='utf-8') as f:
            return json.load(f)['fitur']
    except Exception:
        return FITUR_LAMA


FITUR = _baca_fitur()
HORIZON = list(range(1, 8))

# Rentang kewajaran masukan (Tabel 4.18)
RENTANG = {'DOC': (30, 150, 'DOC'), 'MBW': (0.5, 40, 'MBW'), 'Size': (10, 700, 'Size'),
           'Biomassa': (50, 10000, 'Biomassa'), 'Luas': (500, 10000, 'Luas kolam'),
           'pH_Pagi': (6.5, 9.5, 'pH pagi'), 'pH_Siang': (6.5, 9.5, 'pH siang'),
           'Suhu_Pagi': (22, 34, 'Suhu pagi'), 'Suhu_Siang': (22, 36, 'Suhu siang'),
           'Pakan': (1, 500, 'Pakan hari ini')}
KOLOM_BERKAS = ['Pond'] + FITUR


def _fmt(v):
    return f'{v:g}'.replace('.', ',') if not float(v).is_integer() else f'{int(v)}'


def validasi(nilai: dict, panen: float):
    """Mengembalikan daftar pesan kesalahan (kosong bila semua masukan wajar)."""
    err = []
    for k, (lo, hi, nama) in RENTANG.items():
        if k not in FITUR:
            continue
        v = nilai.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            err.append(f'{nama} tidak boleh kosong.')
        elif not (lo <= v <= hi):
            err.append(f'{nama} = {_fmt(v)} di luar rentang yang diizinkan ({_fmt(lo)}–{_fmt(hi)}).')
    if not (0 <= panen <= 100):
        err.append(f'Persentase panen parsial = {_fmt(panen)} di luar rentang yang diizinkan (0–100).')
    return err


def muat_model():
    ms = {}
    for h in HORIZON:
        m = XGBRegressor()
        m.load_model(BASE / 'model_xgb' / f'xgb_t{h}.json')
        ms[h] = m
    return ms


def prediksi_7_hari(model, nilai: dict):
    x = pd.DataFrame([[float(nilai[k]) for k in FITUR]], columns=FITUR)
    return {h: float(model[h].predict(x)[0]) for h in HORIZON}


# ------------------------------------------------------------------ Fuzzy Mamdani (CoA)
Z = np.linspace(0.40, 1.00, 601)


def tri(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))


def bahu(x, a, b):
    return np.clip((x - a) / (b - a), 0, 1)


MU_PK = tri(Z, 0.80, 0.90, 1.00)
MU_PM = tri(Z, 0.60, 0.725, 0.85)
MU_PB = np.maximum(0, np.minimum(np.minimum((Z - 0.40) / 0.05, 1), (0.65 - Z) / 0.05))

# R1..R9: (prediksi, panen, keluaran)
ATURAN = [('RENDAH', 'KECIL', 'PK'), ('RENDAH', 'SEDANG', 'PM'), ('RENDAH', 'BESAR', 'PB'),
          ('NORMAL', 'KECIL', 'PK'), ('NORMAL', 'SEDANG', 'PM'), ('NORMAL', 'BESAR', 'PB'),
          ('TINGGI', 'KECIL', 'PM'), ('TINGGI', 'SEDANG', 'PB'), ('TINGGI', 'BESAR', 'PB')]


def derajat(prediksi, panen):
    pan = dict(KECIL=float(tri(panen, 0, 10, 20)), SEDANG=float(tri(panen, 15, 30, 45)), BESAR=float(bahu(panen, 40, 60)))
    pre = dict(RENDAH=float(tri(prediksi, 0, 55, 75)), NORMAL=float(tri(prediksi, 65, 90, 115)), TINGGI=float(bahu(prediksi, 105, 125)))
    return pre, pan


def aturan_aktif(prediksi, panen):
    pre, pan = derajat(prediksi, panen)
    return [(i + 1, a, b, o, min(pre[a], pan[b])) for i, (a, b, o) in enumerate(ATURAN) if min(pre[a], pan[b]) > 0]


def faktor_koreksi(prediksi, panen):
    pre, pan = derajat(prediksi, panen)
    alpha = {'PK': 0.0, 'PM': 0.0, 'PB': 0.0}
    for a, b, o in ATURAN:
        alpha[o] = max(alpha[o], min(pre[a], pan[b]))
    if sum(alpha.values()) == 0:
        return 1.0
    agregat = np.maximum.reduce([np.minimum(alpha['PK'], MU_PK), np.minimum(alpha['PM'], MU_PM), np.minimum(alpha['PB'], MU_PB)])
    return float((Z * agregat).sum() / agregat.sum())


def agregasi(prediksi, panen):
    """Untuk visualisasi: kekuatan tiap keluaran (alpha) dan kurva keluaran hasil agregasi (MAX)."""
    pre, pan = derajat(prediksi, panen)
    alpha = {'PK': 0.0, 'PM': 0.0, 'PB': 0.0}
    for a, b, o in ATURAN:
        alpha[o] = max(alpha[o], min(pre[a], pan[b]))
    agregat = np.maximum.reduce([np.minimum(alpha['PK'], MU_PK), np.minimum(alpha['PM'], MU_PM),
                                 np.minimum(alpha['PB'], MU_PB)])
    return alpha, agregat
