"""Sidebar component for StartupPulse AI Dashboard."""

import streamlit as st
import base64
from pathlib import Path

APP_VERSION = "v2.1.0"


def render_sidebar(project_root: Path) -> str:
    with st.sidebar:
        logo_path = project_root / "assets" / "logo.png"
        if logo_path.exists():
            logo_b64 = base64.b64encode(open(str(logo_path), "rb").read()).decode()
            st.markdown(
                f'<div style="display:flex;justify-content:center;padding:4px 0 12px;">'
                f'<img src="data:image/png;base64,{logo_b64}" alt="Logo" '
                f'style="width:40px;height:40px;border-radius:10px;">'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="padding:0 4px 20px;">'
            f'<div style="font-size:16px;font-weight:700;color:#e5e7eb;letter-spacing:-0.02em;margin-bottom:2px;">StartupPulse AI</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">Explainable AI Dashboard</div>'
            f'<span style="display:inline-block;background:rgba(59,130,246,0.12);color:#60a5fa;padding:2px 8px;'
            f'border-radius:4px;font-size:10px;font-weight:600;font-family:monospace;">{APP_VERSION}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'
            'color:#4b5563;margin-bottom:8px;padding-left:4px;">Navigation</div>',
            unsafe_allow_html=True,
        )
        menu = ["Home", "Analyze Review", "Explainability", "Model Metrics", "About"]
        choice = st.radio("Navigation", menu, label_visibility="collapsed")

        st.markdown(
            '<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'
            'color:#4b5563;margin:16px 0 8px 4px;">Stack</div>'
            '<div style="padding:0 4px;">'
            '<div style="font-size:12px;color:#6b7280;padding:3px 0;">DeBERTa-v3</div>'
            '<div style="font-size:12px;color:#6b7280;padding:3px 0;">SHAP</div>'
            '<div style="font-size:12px;color:#6b7280;padding:3px 0;">PyTorch</div>'
            '<div style="font-size:12px;color:#6b7280;padding:3px 0;">HR Analytics</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="padding-top:16px;border-top:1px solid #1e1e30;margin-top:20px;">'
            '<div style="font-size:11px;color:#4b5563;">Python + DeBERTa-v3</div>'
            '<div style="font-size:11px;color:#4b5563;">Powered by SHAP</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    return choice
