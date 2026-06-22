import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from fingerprint import *

st.set_page_config(
    page_title="EE200 Music Identifier",
    layout="wide"
)

# -----------------------------
# Load Database
# -----------------------------
@st.cache_resource
def load_db():
    return pickle.load(open("database.pkl", "rb"))

try:
    DB = load_db()
except:
    st.error("Run build_database.py first.")
    st.stop()

# -----------------------------
# Identification Function
# -----------------------------
def identify(hs):
    votes = Counter()

    for h, tq in hs:
        if h in DB:
            for song, tdb in DB[h]:
                votes[(song, tdb - tq)] += 1

    if not votes:
        return None, 0, votes

    best = votes.most_common(1)[0]
    return best[0][0], best[1], votes


st.title("🎵 EE200 Sonic Signatures Identifier")

page = st.sidebar.radio(
    "Mode",
    ["Single Clip", "Batch Mode"]
)

# =====================================================
# SINGLE CLIP MODE
# =====================================================
if page == "Single Clip":

    up = st.file_uploader(
        "Upload query clip",
        type=["wav", "mp3"]
    )

    noise = st.slider(
        "Add synthetic noise (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

    
   
    st.metric("Noise Level", f"{noise}%")

    if up:

        # Load audio
        original_audio, sr = load_audio(up)
        audio = original_audio.copy()

        signal_power = np.mean(audio**2)

        if noise > 0:

            noise_signal = np.random.randn(len(audio))
            noise_signal = (
                noise_signal *
                (noise / 100) *
                np.std(audio)
            )

            noise_power = np.mean(noise_signal**2)

            snr = 10 * np.log10(
                signal_power / (noise_power + 1e-12)
            )

            st.metric(
                "Estimated SNR",
                f"{snr:.2f} dB"
            )

            audio = audio + noise_signal

            audio = audio / (
                np.max(np.abs(audio)) + 1e-12
            )

        # -----------------------------------
        # Waveform Visualization
        # -----------------------------------
        st.subheader("Waveform")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(audio[:5000])
        ax.set_title("Noisy Audio Waveform")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # -----------------------------------
        # Fingerprinting
        # -----------------------------------
        S = spectrogram(audio)
        P = peaks(S)
        H = hashes(P)

        song, conf, votes = identify(H)

        # -----------------------------------
        # Prediction Section
        # -----------------------------------
        st.subheader("Identification Result")

        col1, col2 = st.columns(2)

        with col1:
            if song:
                st.success(f"Prediction: {song}")
            else:
                st.error("No Match Found")

        with col2:
            st.info(f"Vote Count: {conf}")

        # Confidence Indicator
        if conf > 300:
            st.success("High Confidence Match")
        elif conf > 100:
            st.warning("Moderate Confidence Match")
        else:
            st.error("Low Confidence Match")

        # -----------------------------------
        # Spectrogram
        # -----------------------------------
        st.subheader("Spectrogram")

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            20 * np.log10(S + 1e-6),
            origin="lower",
            aspect="auto"
        )

        ax.set_title("Spectrogram")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

        # -----------------------------------
        # Constellation Map
        # -----------------------------------
        st.subheader("Constellation Map")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.imshow(
            20 * np.log10(S + 1e-6),
            origin="lower",
            aspect="auto"
        )

        ax.scatter(
            [p[1] for p in P],
            [p[0] for p in P],
            s=4
        )

        ax.set_title("Constellation Map")
        st.pyplot(fig)

        # -----------------------------------
        # Offset Histogram
        # -----------------------------------
        if votes:

            vals = [v[1] for v in votes.keys()]

            st.subheader("Offset Histogram")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(vals, bins=60)
            ax.set_title("Offset Histogram")
            st.pyplot(fig)

        # -----------------------------------
        # Top Candidate Songs
        # -----------------------------------
        if votes:

            st.subheader("Top Candidate Matches")

            top_votes = votes.most_common(10)

            names = [
                str(k[0]).split(".")[0]
                for k, v in top_votes
            ]

            counts = [
                v
                for k, v in top_votes
            ]

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.barh(names, counts)

            ax.set_xlabel("Votes")
            ax.set_ylabel("Song")
            ax.set_title("Top Candidate Songs")

            st.pyplot(fig)

        # -----------------------------------
        # Noise Robustness Test
        # -----------------------------------
        st.subheader("Noise Robustness Experiment")

        if st.button("Run Noise Robustness Test"):

            levels = [0, 10, 20, 30, 40, 50]

            results = []

            for level in levels:

                test_audio = original_audio.copy()

                if level > 0:

                    n = (
                        np.random.randn(len(test_audio))
                        * (level / 100)
                        * np.std(test_audio)
                    )

                    test_audio = test_audio + n

                    test_audio = (
                        test_audio
                        / (np.max(np.abs(test_audio)) + 1e-12)
                    )

                pred, confidence, _ = identify(
                    hashes(
                        peaks(
                            spectrogram(test_audio)
                        )
                    )
                )

                results.append([
                    level,
                    pred.split(".")[0]
                    if pred else "Unknown",
                    confidence
                ])

            df = pd.DataFrame(
                results,
                columns=[
                    "Noise (%)",
                    "Prediction",
                    "Confidence"
                ]
            )

            st.dataframe(df)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.plot(
                df["Noise (%)"],
                df["Confidence"],
                marker="o"
            )

            ax.set_xlabel("Noise (%)")
            ax.set_ylabel("Confidence")
            ax.set_title(
                "Fingerprint Robustness vs Noise"
            )

            st.pyplot(fig)

# =====================================================
# BATCH MODE
# =====================================================
else:

    files = st.file_uploader(
        "Upload clips",
        accept_multiple_files=True
    )

    if files:

        rows = []

        for f in files:

            a, sr = load_audio(f)

            pred, _, _ = identify(
                hashes(
                    peaks(
                        spectrogram(a)
                    )
                )
            )

            rows.append([
                f.name,
                pred.split(".")[0]
                if pred else "unknown"
            ])

        df = pd.DataFrame(
            rows,
            columns=[
                "filename",
                "prediction"
            ]
        )

        st.dataframe(df)

        st.download_button(
            "Download results.csv",
            df.to_csv(index=False),
            file_name="results.csv"
        )
