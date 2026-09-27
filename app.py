import streamlit as st
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = "model"
DATASET_PATH = "medical_dataset_merged_clean.csv"
MAX_LENGTH = 256


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    label_names = np.load(
        f"{MODEL_PATH}/label_names.npy",
        allow_pickle=True
    ).tolist()

    return tokenizer, model, device, label_names


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    df = df[
        ["instruction", "input", "output", "topic_set"]
    ].dropna()

    df["instruction"] = df["instruction"].astype(str)
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)
    df["topic_set"] = df["topic_set"].astype(str)

    df["text"] = (
        df["instruction"] + " " + df["input"]
    )

    return df


tokenizer, model, device, label_names = load_model()
df = load_dataset()


def predict_topic(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    return label_names[prediction], confidence


def retrieve_output(query, topic):
    topic_df = df[
        df["topic_set"] == topic
    ].copy()

    if topic_df.empty:
        return None, 0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    documents = topic_df["text"].tolist()

    matrix = vectorizer.fit_transform(
        documents + [query]
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    best_index = similarities.argmax()

    response = topic_df.iloc[best_index]["output"]
    score = similarities[best_index]

    return response, score


st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)


st.title("🏥 Medical Chatbot")

st.warning(
    "Chatbot ini hanya untuk edukasi dan informasi umum, "
    "bukan pengganti diagnosis atau konsultasi dokter."
)

st.write(
    "Masukkan keluhan atau pertanyaan kesehatan Anda."
)


user_input = st.text_area(
    "Pertanyaan",
    placeholder="Contoh: Perut saya kembung, mual, dan tidak nafsu makan",
    height=120
)


if st.button("🔍 Analisis", use_container_width=True):

    if not user_input.strip():

        st.warning(
            "Silakan masukkan pertanyaan terlebih dahulu."
        )

    else:

        with st.spinner("Menganalisis pertanyaan..."):

            topic, confidence = predict_topic(
                user_input
            )

            response, similarity = retrieve_output(
                user_input,
                topic
            )

        st.subheader("Hasil Analisis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Topic",
                topic
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        st.subheader("Jawaban")

        if response:

            st.write(response)

        else:

            st.info(
                "Belum ditemukan jawaban yang sesuai "
                "dalam dataset."
            )