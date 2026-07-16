"""Card components for the dashboard."""


def render_metric_card(value: str, label: str, color: str = "#e5e7eb") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def render_info_card(title: str, body: str, icon_html: str = "") -> str:
    icon_block = f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">{icon_html}<h3 style="margin:0;">{title}</h3></div>' if icon_html else f'<h3>{title}</h3>'
    return (
        f'<div class="info-card">'
        f'{icon_block}'
        f'<p>{body}</p>'
        f'</div>'
    )


def render_sentiment_badge(label: str) -> str:
    css_class = label.lower()
    icon = {"positive": "\u2713", "negative": "\u2717", "neutral": "\u25cf"}.get(css_class, "\u25cf")
    return f'<span class="badge badge-{css_class}">{icon} {label}</span>'


def render_prob_bar(label: str, value: float, color: str) -> str:
    pct = value * 100
    return (
        f'<div class="prob-bar-container">'
        f'<div class="prob-bar-label">'
        f'<span style="color:#9ca3af;font-weight:500;">{label}</span>'
        f'<span style="color:#6b7280;">{pct:.1f}%</span>'
        f'</div>'
        f'<div class="prob-bar-track">'
        f'<div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'</div>'
    )


def render_about_section(icon: str, bg: str, color: str, title: str, body: str) -> str:
    return (
        f'<div class="about-card">'
        f'<div class="about-card-header">'
        f'<div class="about-card-icon" style="background:{bg};color:{color};">{icon}</div>'
        f'<div class="about-card-title">{title}</div>'
        f'</div>'
        f'<div class="about-card-body">{body}</div>'
        f'</div>'
    )


def render_footer() -> str:
    return (
        '<div class="app-footer">'
        '<div class="app-footer-brand">StartupPulse AI</div>'
        '<div class="app-footer-sub">Explainable Aspect-Based Sentiment Analysis</div>'
        '<div class="app-footer-copy">&copy; 2026 StartupPulse AI. MIT License.</div>'
        '</div>'
    )


def render_loading_html(completed_count: int, active_idx: int, progress_pct: int) -> str:
    stages = [
        "Loading DeBERTa-v3 model",
        "Tokenizing review",
        "Running inference",
        "Generating SHAP explanations",
        "Rendering dashboard",
    ]
    rows = ""
    for idx, stage_text in enumerate(stages):
        if idx < completed_count:
            dot_color = "#22c55e"
            text_color = "#9ca3af"
        elif idx == active_idx:
            dot_color = "#3b82f6"
            text_color = "#e5e7eb"
        else:
            dot_color = "#2a2a3d"
            text_color = "#6b7280"
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 10px;font-size:12px;'
            f'color:{text_color};">'
            f'<div style="width:6px;height:6px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>'
            f'<span>{stage_text}</span>'
            f'</div>'
        )
    return (
        f'<div class="card" style="margin-top:12px;">'
        f'<div class="card-header">Processing Pipeline</div>'
        f'{rows}'
        f'<div class="progress-track">'
        f'<div class="progress-fill" style="width:{progress_pct}%;"></div>'
        f'</div>'
        f'</div>'
    )
