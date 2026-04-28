"""Streamlit demo for Turkish text-to-image retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import ProjectConfig
from src.embedder import MultilingualClipEmbedder
from src.retrieval import build_retrieval_index, retrieve_images


st.set_page_config(page_title="Turkce Gorsel Arama", layout="wide")


@st.cache_resource
def load_embedder(model_variant: str) -> MultilingualClipEmbedder:
    config = ProjectConfig()
    config.ensure_directories()
    if model_variant == "finetuned":
        return MultilingualClipEmbedder(
            config,
            text_model_name_or_path=str(config.best_text_checkpoint_dir(config.finetuned_tag)),
            image_model_name_or_path=str(config.best_image_checkpoint_dir(config.finetuned_tag)),
        )
    return MultilingualClipEmbedder(config)


@st.cache_resource
def load_index(split: str, model_variant: str, backend: str):
    config = ProjectConfig()
    config.ensure_directories()
    tag = config.finetuned_tag if model_variant == "finetuned" else None
    return build_retrieval_index(split, config, tag=tag, backend=backend)


def main() -> None:
    config = ProjectConfig()
    config.ensure_directories()

    st.title("Turkce Metin-Gorsel Arama")
    st.write(
        "Turkce bir sorgu girin. Sistem, Flickr8k Turkish veri setindeki en benzer gorselleri "
        "CLIP tabanli ortak embedding uzayinda arar."
    )

    split = st.sidebar.selectbox("Arama split'i", options=["test", "validation", "train"], index=0)
    model_variant = st.sidebar.selectbox("Model", options=["baseline", "finetuned"], index=0)
    backend = st.sidebar.selectbox("Index backend", options=["exact", "ann_lsh"], index=0)
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5, step=1)

    tag = config.finetuned_tag if model_variant == "finetuned" else None
    if not config.split_embedding_path(split, tag=tag).exists():
        st.warning(
            f"{model_variant} icin {split} embedding dosyasi bulunamadi. "
            "Baseline icin `python run_pipeline.py --stage embed`, "
            "fine-tuned model icin `python run_pipeline.py --stage train` calistirin."
        )
        st.stop()

    embedder = load_embedder(model_variant)
    index = load_index(split, model_variant, backend)

    query = st.text_input(
        "Turkce sorgu",
        value="deniz kenarinda kosan kopek",
        placeholder="kanoda kurek ceken kadin",
    )

    if st.button("Ara") and query.strip():
        results_df = retrieve_images(query=query, index=index, embedder=embedder, top_k=top_k)

        st.subheader("Sonuclar")
        for _, row in results_df.iterrows():
            col_image, col_text = st.columns([1, 2])
            with col_image:
                st.image(row["image_path"], use_container_width=True)
            with col_text:
                st.markdown(f"**Rank:** {int(row['rank'])}")
                st.markdown(f"**Similarity:** {row['score']:.4f}")
                st.markdown(f"**Image ID:** {row['image_id']}")
                captions = row["captions"]
                if isinstance(captions, list) and captions:
                    st.markdown("**Captionlar:**")
                    for caption in captions:
                        st.write(f"- {caption}")
            st.divider()


if __name__ == "__main__":
    main()
