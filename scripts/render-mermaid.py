#!/usr/bin/env python3
"""Extract mermaid blocks from markdown, render to SVG, replace with image refs."""

import re
import subprocess
import tempfile
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
DIAGRAMS_DIR = DOCS_DIR / "images" / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

MERMAID_PATTERN = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)

def slugify(text: str, max_len: int = 30) -> str:
    """Create a slug from mermaid content."""
    # Get first meaningful line
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('graph') and not line.startswith('flowchart'):
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', line.lower())[:max_len].strip('-')
            if slug:
                return slug
    return "diagram"

def render_mermaid(content: str, output_path: Path) -> bool:
    """Render mermaid content to SVG."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
        f.write(content)
        f.flush()
        try:
            subprocess.run(
                ['npx', '--yes', '@mermaid-js/mermaid-cli', '-i', f.name, '-o', str(output_path), '-b', 'transparent'],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e.stderr.decode()}")
            return False
        finally:
            Path(f.name).unlink()

def process_file(md_path: Path) -> int:
    """Process a markdown file, return count of replacements."""
    content = md_path.read_text()
    matches = list(MERMAID_PATTERN.finditer(content))

    if not matches:
        return 0

    print(f"\n{md_path.relative_to(DOCS_DIR)}: {len(matches)} diagram(s)")

    # Calculate relative path from md file to diagrams dir
    rel_to_docs = md_path.parent.relative_to(DOCS_DIR)
    depth = len(rel_to_docs.parts)
    img_prefix = "../" * depth + "images/diagrams/"

    replacements = []
    for i, match in enumerate(matches):
        mermaid_content = match.group(1)
        slug = slugify(mermaid_content)

        # Make unique filename based on file + index
        file_slug = md_path.stem.replace('part', 'p')
        svg_name = f"{file_slug}-{i+1}-{slug}.svg"
        svg_path = DIAGRAMS_DIR / svg_name

        print(f"  -> {svg_name}")

        if render_mermaid(mermaid_content, svg_path):
            img_tag = f"![{slug}]({img_prefix}{svg_name})"
            replacements.append((match.group(0), img_tag))
        else:
            print(f"  SKIPPED: render failed")

    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)

    md_path.write_text(content)
    return len(replacements)

def main():
    total = 0
    for md_path in DOCS_DIR.rglob("*.md"):
        total += process_file(md_path)

    print(f"\n=== Rendered {total} diagrams ===")

if __name__ == "__main__":
    main()
