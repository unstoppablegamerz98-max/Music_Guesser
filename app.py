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

@st.cache_resource
def load_db():
    return pickle.load(open("database.pkl", "rb"))

try:
    DB = load_db()
except:
    st.error("Run build_database.py first.")
    st.stop()


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


st.title("EE200 Sonic Signatures Identifier")

page = st.sidebar.radio(
    "Mode",
    ["Single Clip", "Batch Mode", "Database Stats"]
)

# =====================================================
# DATABASE STATS
# =====================================================

if page == "Database Stats":

    st.metric("Unique Hashes", len(DB))

# =====================================================
# SINGLE CLIP MODE
# =====================================================

elif page == "Single Clip":

    up = st.file_uploader(
        "Upload query clip",
        type=["wav", "mp3"]
    )

    noise = st.slider(
        "Add Synthetic Noise (%)",
        0,
        100,
        0
    )

    noise_multiplier = st.slider(
        "Noise Strength Multiplier",
        1.0,
        10.0,
        1.0,
        0.5
    )

    if up:

        audio, sr = load_audio(up)

        # -------------------------------------------------
        # FIND RECOGNITION FAILURE POINT
        # -------------------------------------------------

        if st.button("Find Recognition Failure Point"):

            results = []

            original_pred, _, _ = identify(
                hashes(
                    peaks(
                        spectrogram(audio)
                    )
                )
            )

            failure_noise = None

            for noise_level in range(0, 101, 5):

                noisy = (
                    audio
                    + np.random.randn(len(audio))
                    * (noise_level / 100)
                    * 5
                    * np.std(audio)
                )

                pred, conf, _ = identify(
                    hashes(
                        peaks(
                            spectrogram(noisy)
                        )
                    )
                )

                pred_name = (
                    pred.split(".")[0]
                    if pred
                    else "unknown"
                )

                results.append(
                    [
                        noise_level,
                        pred_name,
                        conf
                    ]
                )

                if (
                    pred != original_pred
                    and failure_noise is None
                ):
                    failure_noise = noise_level

            df_test = pd.DataFrame(
                results,
                columns=[
                    "Noise %",
                    "Prediction",
                    "Votes"
                ]
            )

            st.subheader("Noise Robustness Test")

            st.dataframe(df_test)

            fig, ax = plt.subplots()

            ax.plot(
                df_test["Noise %"],
                df_test["Votes"],
                marker="o"
            )

            ax.set_xlabel("Noise Level (%)")
            ax.set_ylabel("Vote Count")
            ax.set_title(
                "Recognition Robustness vs Noise"
            )

            st.pyplot(fig)

            if failure_noise is not None:

                st.error(
                    f"Recognition first failed at approximately "
                    f"{failure_noise}% noise"
                )

            else:

                st.success(
                    "Recognition remained correct "
                    "up to 100% noise"
                )

        # -------------------------------------------------
        # USER SELECTED NOISE
        # -------------------------------------------------

        if noise > 0:

            audio = (
                audio
                + np.random.randn(len(audio))
                * (noise / 100)
                * noise_multiplier
                * np.std(audio)
            )

            audio = audio / np.max(np.abs(audio))

        # -------------------------------------------------
        # NORMAL IDENTIFICATION
        # -------------------------------------------------

        S = spectrogram(audio)
        P = peaks(S)
        H = hashes(P)

        song, conf, votes = identify(H)

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Prediction: "
                f"{song.split('.')[0] if song else 'Unknown'}"
            )

            st.info(
                f"Vote Count: {conf}"
            )

        # Spectrogram

        fig, ax = plt.subplots()

        im = ax.imshow(
            20 * np.log10(S + 1e-6),
            origin="lower",
            aspect="auto"
        )

        ax.set_title("Spectrogram")

        plt.colorbar(im, ax=ax)

        st.pyplot(fig)

        # Constellation Map

        fig, ax = plt.subplots()

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

        # Offset Histogram

        if votes:

            vals = [
                v[1]
                for v in votes.keys()
            ]

            fig, ax = plt.subplots()

            ax.hist(
                vals,
                bins=60
            )

            ax.set_title(
                "Offset Histogram"
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

            rows.append(
                [
                    f.name,
                    pred.split(".")[0]
                    if pred
                    else "unknown"
                ]
            )

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
