"""Pembaca berkas data harian yang fleksibel (CSV / Excel).

Prinsip: terima sebanyak mungkin bentuk berkas, dan hanya tampilkan error bila data benar-benar tidak sesuai
parameter (kolom wajib tidak ada, kolam tanpa baris lengkap, nilai di luar rentang).

Yang ditangani otomatis:
- format: .xlsx, .xlsm, .xls, .csv, .txt, .tsv (dikenali dari isi berkas, bukan hanya ekstensi)
- Excel berisi banyak sheet: setiap sheet diperiksa, sheet yang berisi kolom wajib dipilih
- judul/baris kosong di atas tabel (baris header dicari otomatis)
- nama kolom beda penulisan: "pH Pagi", "ph_pagi", "Kolam" (= Pond), dst.
- angka berformat Indonesia: "8,5", "1.234,5", "17,38%"
- baris kosong/tak lengkap: dilewati, bukan menyebabkan error
"""
import io
import re

import numpy as np
import pandas as pd

import pipeline_pakan as pp

WAJIB = list(pp.KOLOM_BERKAS)                     # Pond + fitur model
OPSIONAL_ANGKA = ["Pakan_Standar", "Status_Panen", "Persentase_Panen"]
OPSIONAL_TEKS = ["Subset"]

ALIAS = {
    "Pond": ["kolam", "petak", "nokolam", "namakolam", "kodekolam"],
    "DOC": ["dayofculture", "umur", "umurbudidaya", "umurhari"],
    "MBW": ["meanbodyweight", "abw", "bobotrata", "bobotratarata", "bobotrataratag"],
    "Size": ["sizeekorkg", "ekorkg", "ukuran"],
    "Biomassa": ["biomass", "biomassakg"],
    "Luas": ["luaskolam", "luasm2", "area"],
    "pH_Pagi": ["phpagi"],
    "pH_Siang": ["phsiang"],
    "Suhu_Pagi": ["suhupagi", "temppagi", "suhuairpagi"],
    "Suhu_Siang": ["suhusiang", "tempsiang", "suhuairsiang"],
    "Pakan": ["feed", "pakanharian", "pakankg", "pakanharini", "pakanhariinikg"],
    "Pakan_Standar": ["pakanstandar", "fdstd", "fdstandar"],
    "Status_Panen": ["statuspanen", "panenparsial"],
    "Persentase_Panen": ["persentasepanen", "persenpanen", "panenpersen", "persentasepanenparsial"],
    "Subset": ["subsetdata", "datasubset"],
}
CONTOH_NAMA = ("Pond/Kolam, DOC, MBW, Size, Biomassa, Luas, pH_Pagi (atau 'pH Pagi'), pH_Siang, "
               "Suhu_Pagi, Suhu_Siang, Pakan")


class BerkasError(Exception):
    """Berkas tidak dapat dibuka sama sekali (rusak, kosong, format tidak dikenal)."""


def _kunci(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


PETA = {}
for _kanon, _daftar in ALIAS.items():
    PETA[_kunci(_kanon)] = _kanon
    for _a in _daftar:
        PETA[_kunci(_a)] = _kanon
for _k in WAJIB:
    PETA.setdefault(_kunci(_k), _k)


def _kanon(v):
    """Nama kolom baku untuk sebuah header, atau None bila tidak dikenali (mendukung variasi seperti 'Pakan Harian (kg)')."""
    k = _kunci(v)
    if k in PETA:
        return PETA[k]
    if k.startswith("pakan") and not any(x in k for x in ("standar", "std")):
        return "Pakan"
    for kata, nama in (("pagi", "Pagi"), ("siang", "Siang")):
        if kata in k:
            if "ph" in k and "suhu" not in k:
                return f"pH_{nama}"
            if "suhu" in k or "temp" in k:
                return f"Suhu_{nama}"
    for awalan, nama in (("biomassa", "Biomassa"), ("luas", "Luas"), ("mbw", "MBW"), ("size", "Size")):
        if k.startswith(awalan):
            return nama
    return None


def _ke_angka(x):
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    if x is None:
        return np.nan
    t = str(x).strip().replace(" ", "").replace("\u00a0", "")
    if t.lower() in ("", "nan", "none", "-", "na", "n/a", "#n/a", "null"):
        return np.nan
    t = t.rstrip("%")
    if "," in t and "." in t:                       # 1.234,56
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:                                  # 8,5
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _rapikan(raw: pd.DataFrame):
    """Ubah tabel mentah (tanpa header) menjadi tabel dengan nama kolom baku. None bila kosong."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return None
    skor = [sum(1 for v in raw.iloc[i] if _kanon(v)) for i in range(min(len(raw), 30))]
    h = int(np.argmax(skor)) if max(skor) > 0 else 0
    kolom = [_kanon(v) or str(v).strip() for v in raw.iloc[h]]
    df = raw.iloc[h + 1:].copy()
    df.columns = kolom
    df = df.loc[:, ~pd.Index(kolom).duplicated()].reset_index(drop=True)

    for k in [c for c in WAJIB if c != "Pond"] + OPSIONAL_ANGKA:
        if k in df.columns:
            df[k] = df[k].map(_ke_angka)
    if "Subset" in df.columns:
        df["Subset"] = df["Subset"].astype(str).str.strip().str.capitalize()
    if "Pond" in df.columns:
        ada = df["Pond"].notna() & (df["Pond"].astype(str).str.strip() != "")
        df = df[ada].copy()
        angka = df["Pond"].map(_ke_angka)
        if len(df) and angka.notna().all():
            df["Pond"] = angka.astype(int) if (angka % 1 == 0).all() else angka
        else:
            df["Pond"] = df["Pond"].astype(str).str.strip()
    if "DOC" in df.columns:
        df = df[df["DOC"].notna()].copy()
    return df.reset_index(drop=True)


def baca(f):
    """Baca berkas unggahan. Kembalikan {nama_lembar: (DataFrame_rapi, [kolom_wajib_yang_kurang])}."""
    data = f.getvalue() if hasattr(f, "getvalue") else f.read()
    if not data:
        raise BerkasError("Berkas kosong (ukuran 0 byte).")

    lembar_mentah = {}
    if data[:2] == b"PK" or data[:4] == b"\xd0\xcf\x11\xe0":          # xlsx/xlsm  atau  xls lama
        try:
            lembar_mentah = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        except ImportError as e:
            raise BerkasError("Format Excel ini membutuhkan paket tambahan yang belum terpasang "
                              f"({e}). Simpan ulang berkas sebagai .xlsx atau .csv lalu unggah kembali.")
        except Exception as e:
            raise BerkasError("Berkas Excel tidak dapat dibuka (kemungkinan rusak atau diproteksi password). "
                              f"Detail: {type(e).__name__}. Coba simpan ulang sebagai .xlsx atau .csv.")
    else:
        teks = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                teks = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if not teks or not teks.strip():
            raise BerkasError("Isi berkas tidak dapat dibaca sebagai teks/CSV.")
        df_teks = None
        for sep in (None, ";", ",", "\t"):
            try:
                df_teks = pd.read_csv(io.StringIO(teks), sep=sep, engine="python", header=None, dtype=str,
                                      skip_blank_lines=True, on_bad_lines="skip")
                if df_teks.shape[1] > 1:
                    break
            except Exception:
                continue
        if df_teks is None or df_teks.empty:
            raise BerkasError("Isi berkas tidak dikenali sebagai tabel CSV (pemisah kolom tidak terdeteksi).")
        lembar_mentah = {"Data": df_teks}

    hasil = {}
    for nama, raw in lembar_mentah.items():
        df = _rapikan(raw)
        if df is None or df.empty:
            continue
        hasil[str(nama)] = (df, [k for k in WAJIB if k not in df.columns])
    if not hasil:
        raise BerkasError("Berkas tidak berisi data (semua lembar kosong).")
    return hasil


def lembar_valid(hasil):
    return [n for n, (_, kurang) in hasil.items() if not kurang]


def lembar_default(valid):
    for n in valid:
        if "bersih" in n.lower():
            return n
    return valid[0]


def pesan_kolom_kurang(hasil) -> str:
    """Pesan error spesifik bila tidak ada lembar yang memuat semua kolom wajib."""
    terbaik = min(hasil, key=lambda n: len(hasil[n][1]))
    kurang = hasil[terbaik][1]
    dimana = f" (lembar terdekat: '{terbaik}')" if len(hasil) > 1 else ""
    return (f"Kolom wajib tidak ditemukan{dimana}: {', '.join(kurang)}. "
            f"Nama kolom yang dikenali: {CONTOH_NAMA}.")


def baris_lengkap(df: pd.DataFrame, pond) -> pd.DataFrame:
    """Baris kolam tertentu yang semua fitur modelnya terisi angka, urut DOC."""
    d = df[df["Pond"] == pond]
    return d.dropna(subset=["DOC"] + pp.FITUR).sort_values("DOC")


def kolom_bermasalah(df: pd.DataFrame, pond) -> list:
    """Kolom fitur yang kosong/bukan angka pada baris DOC terakhir kolam itu (untuk pesan error)."""
    d = df[df["Pond"] == pond].sort_values("DOC")
    if d.empty:
        return []
    b = d.iloc[-1]
    return [k for k in pp.FITUR if pd.isna(b[k])]
