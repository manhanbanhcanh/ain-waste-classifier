"""Streamlit starter — Project 11 visual waste sorting assistant.

This page uploads an image and shows a preview. It must not classify,
train, or call a hosted model: ``student_core.predict`` is an empty stub
until you implement it yourself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import CLASSES, class_description, predict  # noqa: E402

st.set_page_config(page_title="Visual Waste Sorting Assistant — starter", layout="wide")
st.title("Visual Waste Sorting Assistant — starter")
st.markdown(
    "The **AI core is student work**. This page only uploads and previews "
    "an image; it must not classify, train, or call a hosted model."
)

with st.sidebar:
    st.header("TrashNet classes")
    for cls in CLASSES:
        st.caption(f"**{cls}** — {class_description(cls)}")
    st.divider()
    st.caption(
        "Try an image from `data/sample/<class>/` for a quick offline "
        "preview while you build the model. Known failure examples "
        "(`*_hard_*.png`) are also in there."
    )

uploaded = st.file_uploader("Upload a waste photo", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, use_container_width=True)
    if st.button("Predict class"):
        
        # making a copy of uploaded image to get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.getbuffer())
            path = Path(tmp.name)

        try:
            predict(model=None, image_path=path)
        except NotImplementedError as exc:
            st.warning(f"Core not implemented yet: {exc}")
        finally:
            path.unlink(missing_ok=True) # clean up after predict()
else:
    st.info(
        "Upload an image to preview it here. Prediction is disabled until "
        "you implement `starter/student_core.py::predict`."
    )
