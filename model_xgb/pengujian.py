"""Halaman Pengujian: membandingkan pakan AKTUAL dengan prediksi XGBoost dan PAKAN AKHIR (setelah koreksi fuzzy)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import muat_berkas as mb
import pipeline_pakan as pp

BASE = Path(__file__).parent
OPSIONAL = {"Pakan_Standar": np.nan, "Status_Panen": 0, "Persentase_Panen": 0.0, "Subset": "-"}


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / y) * 100)


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


@st.cache_data(show_spinner=False)
def bangun_hasil(df: pd.DataFrame, _model) -> pd.DataFrame:
    """Untuk setiap hari dasar (t) dan horizon h: prediksi dari data hari t, dibandingkan dengan pakan aktual hari t+h.
    Koreksi fuzzy hanya diterapkan bila hari target adalah hari panen parsial (faktor = 1 pada hari lain)."""
    df = df.copy()
    for c, v in OPSIONAL.items():
        if c not in df.columns:
            df[c] = v
    df["Status_Panen"] = df["Status_Panen"].fillna(0)
    df["Persentase_Panen"] = df["Persentase_Panen"].fillna(0.0)
    tgt = df.dropna(subset=["Pakan"])[["Pond", "DOC", "Pakan", "Pakan_Standar", "Status_Panen", "Persentase_Panen", "Subset"]].copy()
    df = df.dropna(subset=pp.FITUR)          # hari dasar harus punya semua fitur
    baris = []
    for h in pp.HORIZON:
        t = tgt.copy()
        t["DOC"] = t["DOC"] - h
        t = t.rename(columns={"Pakan": "Aktual", "Pakan_Standar": "FD_std", "Status_Panen": "Status_t",
                              "Persentase_Panen": "Panen_t", "Subset": "Subset_t"})
        kolom_dasar = list(dict.fromkeys(["Pond", "DOC", "Pakan", "Subset"] + pp.FITUR))
        m = df[kolom_dasar].merge(t, on=["Pond", "DOC"], how="inner")
        pred = _model[h].predict(m[pp.FITUR])
        faktor = np.array([pp.faktor_koreksi(p, s) if st_ == 1 else 1.0
                           for p, s, st_ in zip(pred, m["Panen_t"], m["Status_t"])])
        baris.append(pd.DataFrame({
            "Pond": m["Pond"].values, "DOC dasar": m["DOC"].values, "Horizon": h, "DOC target": m["DOC"].values + h,
            "Pakan aktual (kg)": m["Aktual"].values, "Prediksi XGBoost (kg)": pred,
            "Panen target (%)": np.where(m["Status_t"].values == 1, m["Panen_t"].values, 0.0),
            "Faktor koreksi": faktor, "Pakan akhir (kg)": pred * faktor,
            "Persistensi (kg)": m["Pakan"].values, "F/D_std (kg)": m["FD_std"].values,
            "Subset dasar": m["Subset"].values, "Subset target": m["Subset_t"].values}))
    out = pd.concat(baris, ignore_index=True)
    out["Selisih (kg)"] = out["Pakan akhir (kg)"] - out["Pakan aktual (kg)"]
    out["Error (%)"] = out["Selisih (kg)"].abs() / out["Pakan aktual (kg)"] * 100
    return out


def _sig(d: pd.DataFrame) -> pd.Series:
    """Tanda pengenal baris (DOC, MBW, Biomassa, Pakan) untuk mencocokkan dengan data latih/uji skripsi."""
    f = d[["DOC", "MBW", "Biomassa", "Pakan"]].astype(float)
    return (f["DOC"].round().astype(int).astype(str) + "|" + f["MBW"].round(2).astype(str) + "|"
            + f["Biomassa"].round(1).astype(str) + "|" + f["Pakan"].round().astype(int).astype(str))


def tandai_subset(df: pd.DataFrame):
    """Berkas unggahan tanpa kolom Subset: tandai baris yang cocok dengan data latih ('Latih'), data uji skripsi ('Uji'),
    atau belum pernah dilihat model ('Baru'). Bila berkas sudah punya Subset Latih/Uji, dipakai apa adanya."""
    if "Subset" in df.columns and df["Subset"].isin(["Latih", "Uji"]).any():
        return df, None
    ref = pd.read_csv(BASE / "data_pengujian.csv")
    latih, uji = set(_sig(ref[ref["Subset"] == "Latih"])), set(_sig(ref[ref["Subset"] == "Uji"]))
    d = df.dropna(subset=["DOC", "MBW", "Biomassa", "Pakan"]).copy()
    sig = _sig(d)
    df = df.copy()
    df["Subset"] = "Baru"
    df.loc[d.index[sig.isin(uji).values], "Subset"] = "Uji"
    df.loc[d.index[sig.isin(latih).values], "Subset"] = "Latih"
    return df, df["Subset"].value_counts().to_dict()


def _ringkas(d, kolom_pred):
    y = d["Pakan aktual (kg)"]
    return dict(MAPE=mape(y, d[kolom_pred]), RMSE=rmse(y, d[kolom_pred]), MAE=mae(y, d[kolom_pred]))


def tampil(model):
    st.title("🧪 Pengujian: Pakan Aktual vs Prediksi")
    st.markdown(
        "Halaman ini membandingkan **pakan aktual** (yang benar-benar diberikan) dengan **prediksi XGBoost** dan "
        "**pakan akhir** (prediksi XGBoost setelah koreksi fuzzy pada hari panen parsial). "
        "Dasarnya: ambil data suatu hari (t), prediksi pakan t+1 sampai t+7, lalu cocokkan dengan pakan aktual pada hari-hari tersebut."
    )

    sumber = st.radio("Sumber data pengujian",
                      ["Data bawaan (dataset skripsi, 5 kolam)", "Unggah berkas sendiri (CSV/Excel berisi pakan aktual)"],
                      horizontal=True)
    if sumber.startswith("Data bawaan"):
        df = pd.read_csv(BASE / "data_pengujian.csv")
        bawaan = True
    else:
        st.caption("Menerima Excel (.xlsx/.xlsm/.xls) dan CSV. Kolom yang dibutuhkan: " + mb.CONTOH_NAMA +
                   ". Kolom opsional: Pakan_Standar, Status_Panen, Persentase_Panen, Subset. Berkas harus mencakup hari-hari "
                   "setelah hari dasar (pakan aktual diambil dari kolom Pakan pada baris DOC+1 sampai DOC+7).")
        f = st.file_uploader("Unggah berkas", type=["xlsx", "xlsm", "xls", "csv", "txt", "tsv"], key="uji_file")
        if f is None:
            st.info("Unggah berkas untuk memulai, atau pilih data bawaan.")
            return
        try:
            lembar = mb.baca(f)
        except mb.BerkasError as e:
            st.error(str(e))
            return
        valid = mb.lembar_valid(lembar)
        if not valid:
            st.error(mb.pesan_kolom_kurang(lembar))
            return
        nama = valid[0]
        if len(valid) > 1:
            nama = st.selectbox("Sheet yang dipakai", valid, index=valid.index(mb.lembar_default(valid)), key="uji_sheet")
        df = lembar[nama][0]
        df, hitung = tandai_subset(df)
        if hitung:
            st.info("Pencocokan otomatis dengan data skripsi: "
                    + ", ".join(f"{v} baris {k.lower()}" for k, v in hitung.items())
                    + ". Baris 'latih' sudah pernah dipakai melatih model (hasilnya cenderung terlalu bagus); "
                      "'uji' dan 'baru' adalah data yang tidak dipakai melatih.")
        bawaan = False

    hasil = bangun_hasil(df, model)
    if hasil.empty:
        st.warning("Tidak ada pasangan hari dasar dan hari target pada data ini (butuh minimal 2 hari berurutan per kolam).")
        return

    tab_a, tab_b, tab_c = st.tabs(["Satu kolam & satu hari prediksi", "Seluruh data uji", "Cara membaca"])

    # ------------------------------------------------------------------ A
    with tab_a:
        c1, c2 = st.columns(2)
        pond = c1.selectbox("Kolam", sorted(hasil["Pond"].unique()), key="uji_pond")
        hp = hasil[hasil["Pond"] == pond]
        pilihan = sorted(hp["DOC dasar"].unique())
        # bawaan: default = hari dasar yang masih punya 7 hari aktual dan berada di data uji
        lengkap = [d for d in pilihan if (hp["DOC dasar"] == d).sum() == len(pp.HORIZON)
                   and hp[hp["DOC dasar"] == d]["Subset target"].isin(["Uji", "Baru"]).all()] or pilihan
        doc = c2.selectbox("Hari dasar (DOC) yang diprediksi dari", pilihan,
                           index=pilihan.index(lengkap[-1]), key="uji_doc",
                           help="Prediksi dibuat dari data pada DOC ini, lalu dibandingkan dengan pakan aktual DOC+1 sampai DOC+7.")
        d = hp[hp["DOC dasar"] == doc].sort_values("Horizon")
        if (d["Subset dasar"] == "Latih").any() or (d["Subset target"] == "Latih").any():
            st.warning("Sebagian data pada pilihan ini termasuk **data latih** (sudah pernah dilihat model). "
                       "Untuk pengujian yang adil, pilih hari dasar yang tergolong data uji (DOC besar pada kolam tersebut).")
        tampil_tab = pd.DataFrame({
            "Horizon": [f"t+{h}" for h in d["Horizon"]],
            "DOC target": d["DOC target"].astype(int).values,
            "Pakan aktual (kg)": d["Pakan aktual (kg)"].values,
            "Prediksi XGBoost (kg)": d["Prediksi XGBoost (kg)"].values,
            "Panen (%)": d["Panen target (%)"].values,
            "Faktor koreksi": d["Faktor koreksi"].values,
            "Pakan akhir (kg)": d["Pakan akhir (kg)"].values,
            "Selisih akhir − aktual (kg)": d["Selisih (kg)"].values,
            "Error (%)": d["Error (%)"].values})
        st.dataframe(tampil_tab.style.format({
            "Pakan aktual (kg)": "{:.1f}", "Prediksi XGBoost (kg)": "{:.1f}", "Panen (%)": "{:.2f}",
            "Faktor koreksi": "{:.4f}", "Pakan akhir (kg)": "{:.1f}", "Selisih akhir − aktual (kg)": "{:+.1f}",
            "Error (%)": "{:.1f}"}), hide_index=True, width="stretch")

        r_x, r_a = _ringkas(d, "Prediksi XGBoost (kg)"), _ringkas(d, "Pakan akhir (kg)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAPE XGBoost", f"{r_x['MAPE']:.2f}%")
        m2.metric("MAPE pakan akhir", f"{r_a['MAPE']:.2f}%")
        m3.metric("RMSE XGBoost", f"{r_x['RMSE']:.2f} kg")
        m4.metric("RMSE pakan akhir", f"{r_a['RMSE']:.2f} kg")
        st.caption(f"Dihitung pada {len(d)} hari (Pond {pond}, hari dasar DOC {int(doc)}). Pada hari tanpa panen parsial, "
                   "pakan akhir = prediksi XGBoost (faktor 1).")

        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(d["DOC target"], d["Pakan aktual (kg)"], "o-", color="#222", lw=2, label="Pakan aktual")
        ax.plot(d["DOC target"], d["Prediksi XGBoost (kg)"], "s--", color="#2a6fb5", label="Prediksi XGBoost")
        ax.plot(d["DOC target"], d["Pakan akhir (kg)"], "^-", color="#c8601a", label="Pakan akhir (setelah fuzzy)")
        for x_, y_, p_ in zip(d["DOC target"], d["Pakan aktual (kg)"], d["Panen target (%)"]):
            if p_ > 0:
                ax.annotate(f"panen {p_:.0f}%", (x_, y_), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="red")
        ax.set_xlabel("DOC target"); ax.set_ylabel("Pakan (kg/hari)"); ax.grid(alpha=.3)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ------------------------------------------------------------------ B
    with tab_b:
        ada_subset = (hasil["Subset dasar"] != "-").any()
        c1, c2, c3 = st.columns(3)
        if ada_subset:
            cakupan = c1.selectbox("Data yang diuji", ["Data uji/baru saja (tidak dipakai melatih)", "Semua data"], key="uji_cakupan")
        else:
            cakupan = "Semua data"
            c1.caption("Berkas tidak punya kolom Subset, sehingga semua baris diuji (model mungkin pernah melihat sebagian data ini).")
        pilih_pond = c2.selectbox("Kolam", ["Semua"] + list(sorted(hasil["Pond"].unique())), key="uji_pond_b")
        pilih_h = c3.selectbox("Horizon", ["Semua"] + [f"t+{h}" for h in pp.HORIZON], key="uji_h_b")

        u = hasil.copy()
        if cakupan.startswith("Data uji"):
            u = u[u["Subset dasar"].isin(["Uji", "Baru"]) & u["Subset target"].isin(["Uji", "Baru"])]
        if pilih_pond != "Semua":
            u = u[u["Pond"] == pilih_pond]
        if pilih_h != "Semua":
            u = u[u["Horizon"] == int(pilih_h[2:])]
        if u.empty:
            st.warning("Tidak ada data untuk pilihan ini.")
        else:
            baris_r = {"Pakan akhir (hybrid)": _ringkas(u, "Pakan akhir (kg)"),
                       "Prediksi XGBoost": _ringkas(u, "Prediksi XGBoost (kg)"),
                       "Persistensi": _ringkas(u, "Persistensi (kg)")}
            u_fd = u.dropna(subset=["F/D_std (kg)"])
            if len(u_fd):
                baris_r["F/D_std"] = _ringkas(u_fd, "F/D_std (kg)")
            ringkas = pd.DataFrame(baris_r).T
            ringkas.columns = ["MAPE (%)", "RMSE (kg)", "MAE (kg)"]
            st.markdown(f"**Ringkasan akurasi** (n = {len(u)} pasang prediksi–aktual)")
            st.dataframe(ringkas.style.format("{:.2f}"), width="stretch")

            baris_h = []
            for h_, g in u.groupby("Horizon"):
                y_ = g["Pakan aktual (kg)"]
                baris_h.append({"Horizon": f"t+{h_}", "n": len(g),
                                "MAPE XGBoost (%)": mape(y_, g["Prediksi XGBoost (kg)"]),
                                "MAPE pakan akhir (%)": mape(y_, g["Pakan akhir (kg)"]),
                                "RMSE XGBoost (kg)": rmse(y_, g["Prediksi XGBoost (kg)"]),
                                "RMSE pakan akhir (kg)": rmse(y_, g["Pakan akhir (kg)"])})
            per_h = pd.DataFrame(baris_h)
            st.dataframe(per_h.style.format({**{c: "{:.2f}" for c in per_h.columns if c not in ("Horizon", "n")}, "n": "{:.0f}"}),
                         hide_index=True, width="stretch")

            # grafik aktual vs prediksi sepanjang waktu (satu kolam, satu horizon)
            st.markdown("**Grafik pakan aktual vs prediksi sepanjang periode uji**")
            g1, g2 = st.columns(2)
            gp = g1.selectbox("Kolam untuk grafik", sorted(u["Pond"].unique()), key="uji_gp")
            gh = g2.selectbox("Horizon untuk grafik", [f"t+{h}" for h in pp.HORIZON], key="uji_gh")
            gu = hasil[(hasil["Pond"] == gp) & (hasil["Horizon"] == int(gh[2:]))]
            if cakupan.startswith("Data uji"):
                gu = gu[gu["Subset dasar"].isin(["Uji", "Baru"]) & gu["Subset target"].isin(["Uji", "Baru"])]
            gu = gu.sort_values("DOC target")
            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.plot(gu["DOC target"], gu["Pakan aktual (kg)"], "o-", color="#222", ms=4, lw=1.8, label="Pakan aktual")
            ax.plot(gu["DOC target"], gu["Prediksi XGBoost (kg)"], "--", color="#2a6fb5", lw=1.4, label="Prediksi XGBoost")
            ax.plot(gu["DOC target"], gu["Pakan akhir (kg)"], "-", color="#c8601a", lw=1.6, label="Pakan akhir (setelah fuzzy)")
            pn = gu[gu["Panen target (%)"] > 0]
            ax.plot(pn["DOC target"], pn["Pakan aktual (kg)"], "o", color="red", ms=8, mfc="none", label="Hari panen parsial")
            ax.set_xlabel("DOC target"); ax.set_ylabel("Pakan (kg/hari)"); ax.grid(alpha=.3)
            ax.set_title(f"Pond {gp}, horizon {gh}", fontsize=10)
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.download_button("Unduh rincian pengujian (CSV)", u.to_csv(index=False), "rincian_pengujian.csv", "text/csv")

    # ------------------------------------------------------------------ C
    with tab_c:
        st.markdown("""
**Istilah**
- **Pakan aktual**: pakan yang benar-benar diberikan pada hari target (kolom `Pakan` pada baris DOC+h).
- **Prediksi XGBoost**: pakan hari target yang diprediksi dari data hari dasar (t).
- **Pakan akhir**: prediksi XGBoost × faktor koreksi fuzzy. Faktor hanya ≠ 1 pada hari panen parsial; pada hari lain pakan akhir = prediksi XGBoost.
- **Selisih** = pakan akhir − pakan aktual (positif = terlalu banyak, negatif = kurang). **Error (%)** = |selisih| ÷ aktual × 100.

**Ukuran akurasi**
- **MAPE** = (100 ÷ n) · Σ |aktual − prediksi| ÷ aktual
- **RMSE** = √( (1 ÷ n) · Σ (aktual − prediksi)² )
- **MAE** = (1 ÷ n) · Σ |aktual − prediksi|

**Pembanding**: *Persistensi* (menyalin pakan hari dasar) dan *F/D_std* (pakan standar program tambak).

**Memakai untuk periode ke depan (pemakaian nyata)**
1. Hari ini: di halaman *Prediksi ke depan*, isi data kolam lalu tekan **Prediksi Pakan**. Simpan tabel rekomendasi (tombol unduh CSV).
2. Setelah 7 hari berlalu, catat pakan yang sebenarnya diberikan, lalu unggah data hariannya di halaman ini (*Unggah berkas sendiri*).
3. Aplikasi menyandingkan pakan aktual dengan pakan akhir dan menghitung MAPE serta RMSE.
""")
