import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import pipeline_pakan as pp

BASE = Path(__file__).parent
st.set_page_config(page_title="Rekomendasi Pakan Udang Vannamei", page_icon="🦐", layout="wide")


@st.cache_resource
def model():
    return pp.muat_model()


LUAS = {2: 2500, 4: 2500, 7: 2500, 8: 2500, 9: 2500}
AWAL = dict(DOC=88, MBW=20.65, Size=49.1, Biomassa=3150.2, Luas=2500, pH_Pagi=8.0, pH_Siang=8.5, Suhu_Pagi=27.0, Suhu_Siang=28.0, Pakan=94.0)

# ------------------------------------------------------------------ KETERANGAN FITUR
KET = {
    'DOC': ("DOC – umur budidaya (hari)",
            "Day of Culture: umur budidaya udang dihitung sejak hari tebar benur. DOC 88 berarti udang sudah dipelihara 88 hari."),
    'MBW': ("MBW – bobot rata-rata udang (g)",
            "Mean Body Weight: bobot rata-rata seekor udang hasil sampling, dalam gram."),
    'Size': ("Size – jumlah ekor per kg",
             "Banyaknya ekor udang dalam 1 kg. Makin besar udang, makin kecil angkanya (Size ≈ 1000 ÷ MBW)."),
    'Biomassa': ("Biomassa – total berat udang (kg)",
                 "Perkiraan total berat seluruh udang yang hidup di kolam."),
    'Luas': ("Luas kolam (m²)",
             "Luas permukaan kolam. Bersama biomassa dipakai untuk menghitung kepadatan udang (biomassa ÷ luas)."),
    'pH_Pagi': ("pH pagi – keasaman air",
                "Derajat keasaman air kolam yang diukur pada pagi hari."),
    'pH_Siang': ("pH siang – keasaman air",
                 "Derajat keasaman air kolam yang diukur pada siang hari."),
    'Suhu_Pagi': ("Suhu pagi – air kolam (°C)",
                  "Suhu air kolam yang diukur pada pagi hari."),
    'Suhu_Siang': ("Suhu siang – air kolam (°C)",
                   "Suhu air kolam yang diukur pada siang hari."),
    'Pakan': ("Pakan hari ini (kg)",
              "Total pakan yang diberikan pada hari DOC di atas (satu hari penuh), dalam kilogram. Menjadi dasar prediksi pakan hari-hari berikutnya."),
}
KET_PANEN = ("Hari panen parsial yang direncanakan",
             "t+n berarti n hari setelah hari input. Contoh: jika hari input DOC 88, maka t+3 = DOC 91. Koreksi fuzzy hanya diterapkan pada hari ini.")
KET_PERSEN = "Persentase populasi udang yang dipanen sebagian pada hari tersebut (panen parsial). Isi 0 jika tidak ada rencana panen."

# ------------------------------------------------------------------ SIDEBAR
sb = st.sidebar
halaman = sb.radio("Halaman", ["🔮 Prediksi ke depan", "🧪 Pengujian: aktual vs prediksi"])
if halaman.startswith("🧪"):
    import pengujian
    pengujian.tampil(model())
    st.stop()
sb.divider()
sb.header("Panel Masukan")
mode = sb.radio("Mode masukan", ["Input manual", "Unggah berkas (CSV/Excel)"])
nilai, pond, siap = {}, None, True
with sb.expander("ℹ️ Keterangan istilah / fitur"):
    for _k in pp.FITUR:
        st.markdown(f"**{KET[_k][0]}**  \n{KET[_k][1]}")

if mode == "Input manual":
    pond = sb.selectbox("Kolam (Pond)", [2, 4, 7, 8, 9])
    c1, c2 = sb.columns(2)
    nilai['DOC'] = c1.number_input(KET['DOC'][0], value=AWAL['DOC'], step=1, help=KET['DOC'][1])
    nilai['MBW'] = c2.number_input(KET['MBW'][0], value=AWAL['MBW'], step=0.1, format="%.2f", help=KET['MBW'][1])
    c1, c2 = sb.columns(2)
    nilai['Size'] = c1.number_input(KET['Size'][0], value=AWAL['Size'], step=0.1, help=KET['Size'][1])
    nilai['Biomassa'] = c2.number_input(KET['Biomassa'][0], value=AWAL['Biomassa'], step=1.0, help=KET['Biomassa'][1])
    nilai['Luas'] = sb.number_input(KET['Luas'][0], value=LUAS[pond], step=100, help=KET['Luas'][1])
    c1, c2 = sb.columns(2)
    nilai['pH_Pagi'] = c1.number_input(KET['pH_Pagi'][0], value=AWAL['pH_Pagi'], step=0.1, help=KET['pH_Pagi'][1])
    nilai['pH_Siang'] = c2.number_input(KET['pH_Siang'][0], value=AWAL['pH_Siang'], step=0.1, help=KET['pH_Siang'][1])
    c1, c2 = sb.columns(2)
    nilai['Suhu_Pagi'] = c1.number_input(KET['Suhu_Pagi'][0], value=AWAL['Suhu_Pagi'], step=0.5, help=KET['Suhu_Pagi'][1])
    nilai['Suhu_Siang'] = c2.number_input(KET['Suhu_Siang'][0], value=AWAL['Suhu_Siang'], step=0.5, help=KET['Suhu_Siang'][1])
    if 'Pakan' in pp.FITUR:
        nilai['Pakan'] = sb.number_input(KET['Pakan'][0], value=AWAL['Pakan'], step=1.0, help=KET['Pakan'][1])
else:
    _t = pd.read_csv(BASE / "contoh_data_harian.csv").head(3)
    for _k in pp.KOLOM_BERKAS:
        if _k not in _t.columns:
            _t[_k] = ""          # kolom baru (mis. Pakan) dikosongkan, diisi pengguna
    templat = _t[pp.KOLOM_BERKAS].to_csv(index=False)
    sb.download_button("Unduh templat CSV", templat, "templat_data_harian.csv", "text/csv")
    f = sb.file_uploader("Unggah berkas data harian", type=["csv", "xlsx"])
    err_berkas = None
    if f is not None:
        try:
            df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
            kurang = [k for k in pp.KOLOM_BERKAS if k not in df.columns]
            if kurang:
                err_berkas = "Kolom tidak lengkap: " + ", ".join(kurang) + "."
            else:
                pond = sb.selectbox("Kolam pada berkas", sorted(df['Pond'].unique()))
                baris = df[df['Pond'] == pond].sort_values('DOC').iloc[-1]
                nilai = {k: float(baris[k]) for k in pp.FITUR}
                sb.info(f"Data terakhir kolam {pond}: DOC {int(baris['DOC'])}")
        except Exception:
            err_berkas = "Berkas tidak dapat dibaca. Gunakan templat CSV/Excel yang disediakan."
    else:
        err_berkas = "Unggah berkas data harian terlebih dahulu."
    if err_berkas:
        siap = False

sb.divider()
sb.subheader("Rencana Panen Parsial")
hari = sb.selectbox(KET_PANEN[0], [f"t+{h}" for h in pp.HORIZON], index=2, help=KET_PANEN[1])
panen = sb.number_input("Persentase panen parsial (%)", value=17.38, step=0.5, help=KET_PERSEN)
tombol = sb.button("Prediksi Pakan", type="primary", width="stretch")

# ------------------------------------------------------------------ UTAMA
st.title("🦐 Sistem Rekomendasi Pakan Udang Vannamei")
st.caption("Hybrid XGBoost – Fuzzy Logic | prediksi pakan harian 7 hari ke depan dengan koreksi saat panen parsial")

if mode != "Input manual" and not siap:
    st.warning(err_berkas)

errors = pp.validasi(nilai, panen) if siap else []
for e in errors:
    st.error(e)

if tombol and siap and not errors:
    h_t = int(hari.split("+")[1])
    pred = pp.prediksi_7_hari(model(), nilai)
    doc0 = int(nilai['DOC'])
    baris, final = [], {}
    for h in pp.HORIZON:
        pn = float(panen) if h == h_t else 0.0
        f_ = pp.faktor_koreksi(pred[h], pn) if pn > 0 else 1.0
        final[h] = pred[h] * f_
        baris.append({"Hari": f"t+{h}", "DOC target": doc0 + h, "Prediksi XGBoost (kg)": round(pred[h], 2),
                      "Panen parsial (%)": f"{pn:.2f}" if pn > 0 else "-", "Faktor koreksi": round(f_, 4),
                      "Pakan akhir (kg)": round(final[h], 2)})
    tab_df = pd.DataFrame(baris)

    st.subheader(f"Hasil untuk Pond {pond} (data DOC {doc0}) – hari target {hari} (DOC {doc0 + h_t})")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediksi XGBoost", f"{pred[h_t]:.1f} kg")
    faktor_t = pp.faktor_koreksi(pred[h_t], panen) if panen > 0 else 1.0
    m2.metric("Faktor koreksi fuzzy", f"{faktor_t:.3f}")
    m3.metric("Pakan akhir (rekomendasi)", f"{final[h_t]:.1f} kg", delta=f"{final[h_t] - pred[h_t]:.1f} kg dari XGBoost", delta_color="off")

    t1, t2, t3, t5, t4 = st.tabs(["Grafik 7 hari", "Tabel rekomendasi", "Detail koreksi fuzzy", "Akurasi model", "Tentang model"])
    with t1:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        x = [f"t+{h}" for h in pp.HORIZON]
        ax.plot(x, [pred[h] for h in pp.HORIZON], "o-", color="#2a6fb5", lw=2, label="Prediksi XGBoost")
        ax.plot(x, [final[h] for h in pp.HORIZON], "s--", color="#c8601a", lw=2, label="Pakan akhir (setelah koreksi fuzzy)")
        if panen > 0:
            ax.axvline(h_t - 1, color="#cc0000", ls=":", lw=1.5)
            ax.text(h_t - 1 + 0.05, ax.get_ylim()[1], "panen parsial", color="#cc0000", va="top")
        ax.set_ylabel("Pakan harian (kg)"); ax.grid(alpha=.3); ax.legend(loc="lower right")
        st.pyplot(fig); plt.close(fig)
    with t2:
        st.dataframe(tab_df, hide_index=True, width="stretch")
        st.download_button("Unduh tabel (CSV)", tab_df.to_csv(index=False), "rekomendasi_pakan.csv", "text/csv")
        st.caption("Ini prediksi untuk hari-hari ke depan, sehingga pakan aktualnya belum ada. Setelah hari-hari tersebut berlalu, "
                   "bandingkan dengan pakan aktual di halaman **🧪 Pengujian: aktual vs prediksi** (menu Halaman di panel kiri).")
    with t3:
        st.markdown(
            "**Apa fungsi koreksi fuzzy?** Pada hari panen parsial, jumlah udang di kolam berkurang sehingga "
            "kebutuhan pakan ikut turun. XGBoost belum memperhitungkan rencana panen, maka hasilnya dikalikan "
            "**faktor koreksi** (0,40 – 1,00) dari sistem fuzzy Mamdani. Faktor = 1 berarti tidak ada potongan.")
        with st.expander("📘 Cara kerja & keterangan istilah fuzzy"):
            st.markdown("""
**Empat langkah perhitungan**
1. **Fuzzifikasi**: prediksi XGBoost (kg) dan persentase panen diubah menjadi derajat keanggotaan **μ** (0–1) pada tiap himpunan.
2. **Evaluasi aturan** (9 aturan JIKA–MAKA): kekuatan tiap aturan **α** = nilai terkecil (MIN, operator DAN) dari dua μ yang terlibat.
3. **Agregasi**: kurva keluaran dari semua aturan digabung dengan nilai terbesar (MAX).
4. **Defuzzifikasi** *Centroid of Area* (**CoA**): titik berat kurva gabungan menjadi satu angka, yaitu faktor koreksi.

**Istilah**
- **μ (derajat keanggotaan)**: seberapa besar suatu nilai termasuk dalam himpunan (0 = tidak termasuk, 1 = termasuk penuh).
- **α (kekuatan aturan)**: seberapa kuat sebuah aturan berlaku; aturan dengan α = 0 tidak aktif.
- **PK / PM / PB (keluaran)**: besarnya potongan pakan, yaitu **P**otongan **K**ecil, **M**oderat, dan **B**esar.
- **Pakan akhir** = prediksi XGBoost × faktor koreksi.
""")
            st.markdown("**Himpunan fuzzy yang dipakai**")
            st.dataframe(pd.DataFrame([
                ("Prediksi XGBoost (kg)", "RENDAH", "0 – 75 (puncak 55)"),
                ("Prediksi XGBoost (kg)", "NORMAL", "65 – 115 (puncak 90)"),
                ("Prediksi XGBoost (kg)", "TINGGI", "mulai 105, penuh di 125"),
                ("Persentase panen (%)", "KECIL", "0 – 20 (puncak 10)"),
                ("Persentase panen (%)", "SEDANG", "15 – 45 (puncak 30)"),
                ("Persentase panen (%)", "BESAR", "mulai 40, penuh di 60"),
                ("Faktor koreksi (keluaran)", "PK – potongan kecil", "0,80 – 1,00 (puncak 0,90)"),
                ("Faktor koreksi (keluaran)", "PM – potongan moderat", "0,60 – 0,85 (puncak 0,725)"),
                ("Faktor koreksi (keluaran)", "PB – potongan besar", "0,40 – 0,65 (pusat 0,525)"),
            ], columns=["Variabel", "Himpunan", "Rentang"]), hide_index=True, width="stretch")
            st.markdown("**Sembilan aturan (JIKA prediksi … DAN panen … MAKA faktor …)**")
            aktif = {r[0] for r in pp.aturan_aktif(pred[h_t], panen)} if panen > 0 else set()
            st.dataframe(pd.DataFrame([(i + 1, x, y, z, "✓" if (i + 1) in aktif else "")
                                       for i, (x, y, z) in enumerate(pp.ATURAN)],
                                      columns=["Aturan", "Jika prediksi", "dan panen", "maka faktor", "Aktif sekarang"]),
                         hide_index=True, width="stretch")
        if panen > 0:
            pre, pan = pp.derajat(pred[h_t], panen)
            a, b = st.columns(2)
            a.markdown("**Derajat keanggotaan prediksi XGBoost**")
            a.dataframe(pd.DataFrame({"μ": pre}).round(4).style.format("{:.4f}"), width="stretch")
            b.markdown("**Derajat keanggotaan panen parsial**")
            b.dataframe(pd.DataFrame({"μ": pan}).round(4).style.format("{:.4f}"), width="stretch")
            st.markdown("**Aturan yang aktif**")
            ra = pp.aturan_aktif(pred[h_t], panen)
            st.dataframe(pd.DataFrame(ra, columns=["Aturan", "Jika prediksi", "dan panen", "maka faktor", "α"])
                         .style.format({"α": "{:.4g}"}), hide_index=True, width="stretch")
            st.success(f"Faktor koreksi hasil defuzzifikasi (CoA) = {pp.faktor_koreksi(pred[h_t], panen):.5f}")
            fk = pp.faktor_koreksi(pred[h_t], panen)
            st.caption(f"Artinya pakan pada hari panen (t+{h_t}) dikurangi sekitar {(1 - fk) * 100:.1f}% dari prediksi XGBoost: "
                       f"{pred[h_t]:.1f} kg × {fk:.4f} = {pred[h_t] * fk:.1f} kg.")
            alpha, agregat = pp.agregasi(pred[h_t], panen)
            fig, ax = plt.subplots(figsize=(8, 3.6))
            for nm, mu, w in [("PK", pp.MU_PK, "#4a7c2f"), ("PM", pp.MU_PM, "#c8601a"), ("PB", pp.MU_PB, "#2a6fb5")]:
                ax.plot(pp.Z, mu, color=w, lw=1, alpha=.6, label=f"{nm} (α = {alpha[nm]:.3f})")
            ax.fill_between(pp.Z, agregat, color="#999", alpha=.35, label="Hasil agregasi (MAX)")
            ax.axvline(fk, color="red", ls="--", lw=1.5, label=f"Titik berat (CoA) = {fk:.4f}")
            ax.set_xlabel("Faktor koreksi"); ax.set_ylabel("μ"); ax.set_ylim(0, 1.05)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Tidak ada panen parsial, sehingga faktor koreksi = 1,0000.")
    with t5:
        f_metrik = BASE / "model_xgb" / "metrik_uji.csv"
        if not f_metrik.exists():
            st.info("Berkas metrik uji (model_xgb/metrik_uji.csv) tidak ditemukan.")
        else:
            mt = pd.read_csv(f_metrik)
            st.markdown("**Akurasi pada data uji** (bukan pada data yang sedang Anda masukkan). "
                        "Aktual = pakan sebenarnya pada hari target; semakin kecil MAPE/RMSE semakin baik.")
            kolom = {"Horizon": "Horizon", "n": "Jumlah uji",
                     "MAPE_XGB": "MAPE XGBoost (%)", "RMSE_XGB": "RMSE XGBoost (kg)",
                     "MAPE_Hybrid": "MAPE Hybrid (%)", "RMSE_Hybrid": "RMSE Hybrid (kg)",
                     "MAPE_FD": "MAPE F/D_std (%)", "RMSE_FD": "RMSE F/D_std (kg)",
                     "MAPE_Persistensi": "MAPE Persistensi (%)", "RMSE_Persistensi": "RMSE Persistensi (kg)",
                     "MAPE_XGB_tanpa_Pakan": "MAPE XGBoost tanpa Pakan (%)",
                     "RMSE_XGB_tanpa_Pakan": "RMSE XGBoost tanpa Pakan (kg)"}
            tampil = mt[[k for k in kolom if k in mt.columns]].rename(columns=kolom)
            fmt = {c: "{:.2f}" for c in tampil.columns if c not in ("Horizon", "Jumlah uji")}
            st.dataframe(tampil.style.format(fmt), hide_index=True, width="stretch")

            per_h = mt[mt["Horizon"] != "Gabungan"]
            fig, ax = plt.subplots(figsize=(10, 3.6))
            lebar, xs = 0.27, np.arange(len(per_h))
            for i, (k, nama, w) in enumerate([("MAPE_XGB", "XGBoost", "#2a6fb5"),
                                              ("MAPE_FD", "F/D_std (standar)", "#4a7c2f"),
                                              ("MAPE_Persistensi", "Persistensi (pakan hari ini)", "#c8601a")]):
                ax.bar(xs + (i - 1) * lebar, per_h[k], lebar, label=nama, color=w)
            ax.set_xticks(xs); ax.set_xticklabels(per_h["Horizon"])
            ax.set_ylabel("MAPE (%)"); ax.grid(axis="y", alpha=.3); ax.legend(ncol=3, loc="upper left")
            st.pyplot(fig); plt.close(fig)
            g_ = mt[mt["Horizon"] == "Gabungan"].iloc[0]
            st.caption(f"Gabungan semua horizon (n = {int(g_['n'])}): MAPE XGBoost {g_['MAPE_XGB']:.2f}% "
                       f"| F/D_std {g_['MAPE_FD']:.2f}% | Persistensi {g_['MAPE_Persistensi']:.2f}%. "
                       "Data uji: 5 kolam, pembagian latih/uji 80:20 berdasarkan waktu. "
                       "Hybrid hanya berbeda dari XGBoost pada hari panen parsial.")
    with t4:
        st.markdown(f"""
- **Model:** tujuh model XGBoost (t+1 sampai t+7), masing-masing dilatih pada data latih.
- **Fitur:** {", ".join(pp.FITUR)}.
- **Hiperparameter:** 500 pohon, learning rate 0,05, kedalaman maksimum 3, subsample 0,8, colsample 0,8.
- **Koreksi fuzzy:** Mamdani, 9 aturan, defuzzifikasi Center of Area (CoA); hanya aktif pada hari panen parsial.
- **Catatan:** aplikasi ini prototipe alat bantu keputusan, bukan pengganti penilaian teknisi tambak.""")
        st.markdown("**Keterangan fitur masukan**")
        st.dataframe(pd.DataFrame([(KET[k][0], KET[k][1]) for k in pp.FITUR], columns=["Fitur", "Keterangan"]),
                     hide_index=True, width="stretch")
else:
    if True:
        st.info("Isi data pada panel kiri lalu tekan **Prediksi Pakan**. Nilai awal adalah contoh data Pond 2 pada DOC 88.")
