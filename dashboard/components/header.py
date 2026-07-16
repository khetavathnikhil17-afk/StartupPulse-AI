"""Page header component."""


def render_page_header(title: str, description: str) -> str:
    return (
        f'<div class="page-header">'
        f'<h1>{title}</h1>'
        f'<p>{description}</p>'
        f'</div>'
    )
