"""Chart helper components."""


def render_horizontal_bar(label: str, pct: float, color: str) -> str:
    return (
        f'<div style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-size:12px;font-weight:500;color:#9ca3af;">{label}</span>'
        f'<span style="font-size:12px;font-weight:600;color:{color};">{pct:.0f}%</span>'
        f'</div>'
        f'<div style="height:8px;background:#1e1e30;border-radius:4px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;background:{color};border-radius:4px;"></div>'
        f'</div>'
        f'</div>'
    )


def render_circular_gauge(pct: float, color: str = "#3b82f6") -> str:
    r = 42
    circ = 2 * 3.14159 * r
    offset = circ * (1 - pct / 100)
    return (
        f'<svg viewBox="0 0 100 100" style="width:80px;height:80px;">'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="#1e1e30" stroke-width="10"/>'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="10" '
        f'stroke-linecap="round" stroke-dasharray="{circ}" stroke-dashoffset="{offset}" '
        f'transform="rotate(-90 50 50)"/>'
        f'<text x="50" y="50" text-anchor="middle" dominant-baseline="central" '
        f'font-family="Inter, sans-serif" font-size="14" font-weight="700" '
        f'fill="{color}">{pct}%</text>'
        f'</svg>'
    )


def render_svg_line_chart() -> str:
    return (
        '<svg viewBox="0 0 400 100" class="chart-area" preserveAspectRatio="none" '
        'style="width:100%;height:100px;">'
        '<defs>'
        '<linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.15"/>'
        '<stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>'
        '</linearGradient>'
        '</defs>'
        '<path d="M0,70 C40,25 80,80 120,45 S200,15 240,50 S320,75 360,28 L400,32 L400,100 L0,100 Z" '
        'fill="url(#areaGrad)"/>'
        '<path d="M0,70 C40,25 80,80 120,45 S200,15 240,50 S320,75 360,28 L400,32" '
        'fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round"/>'
        '<circle cx="240" cy="50" r="3" fill="#3b82f6" stroke="#13131f" stroke-width="2"/>'
        '</svg>'
    )


def render_svg_bar_chart() -> str:
    bars = [
        ("Mon", 45), ("Tue", 65), ("Wed", 50),
        ("Thu", 80), ("Fri", 95), ("Sat", 35), ("Sun", 25),
    ]
    bar_html = ""
    x = 20
    for day, height in bars:
        bar_h = height * 0.8
        y = 100 - bar_h
        is_active = day == "Fri"
        color = "#3b82f6" if is_active else "#2a2a3d"
        bar_w = "14" if is_active else "10"
        bar_html += (
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'rx="3" fill="{color}"/>'
            f'<text x="{x + int(bar_w) // 2}" y="112" text-anchor="middle" '
            f'font-size="9" fill="#6b7280" font-family="Inter, sans-serif">{day}</text>'
        )
        x += 50
    return (
        f'<svg viewBox="0 0 380 120" class="chart-area" preserveAspectRatio="none" '
        f'style="width:100%;height:100px;">'
        f'{bar_html}'
        f'</svg>'
    )
