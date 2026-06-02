import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUTPUT_DIR     = Path("outputs")
RESULTS_PATH   = OUTPUT_DIR / "results.csv"
SAMPLES_DIR    = OUTPUT_DIR / "cluster_samples"
HTML_OUT       = OUTPUT_DIR / "cluster_map.html"
 
SAMPLES_PER_CLUSTER = 9    # images shown in the sample grid per cluster
GRID_IMG_SIZE       = 200  # px per image in the grid

CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#264653", "#F4A261",
    "#A8DADC", "#457B9D", "#6A4C93", "#F77F00", "#4CC9F0",
    "#80B918", "#FF6B6B", "#C77DFF", "#06D6A0", "#FFB703",
]


def make_scatter(df: pd.DataFrame) -> go.Figure:
    """Interactive UMAP scatter — one point per page, colored by cluster."""
    df = df.copy()
    df["cluster_str"] = df["cluster"].astype(str)
    df["label"] = df["cluster"].apply(
        lambda c: "Noise" if c == -1 else f"Cluster {c}"
    )
    df["hover"] = df.apply(
        lambda r: f"<b>{r['manga_title']}</b><br>{Path(r['image_path']).name}<br>Cluster {r['cluster']}",
        axis=1
    )

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="label",
        hover_name="hover",
        title="Manga Art Style Clusters (UMAP projection)",
        color_discrete_sequence=CLUSTER_COLORS,
        labels={"label": "Cluster"},
        template="plotly_white",
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.75, line=dict(width=0)),
        hovertemplate="%{hovertext}<extra></extra>",
    )

    fig.update_layout(
        width=1000,
        height=700,
        title_font_size=18,
        legend_title_text="Cluster",
        font=dict(family="monospace"),
    )
    return fig


def make_cluster_grid(df: pd.DataFrame, cluster_id: int, n: int = 9) -> Image.Image:
    """
    Take up to n sample pages from a cluster and tile them into a grid image.
    Adds the manga title as a caption under each panel.
    """
    subset = df[df["cluster"] == cluster_id].sample(
        min(n, len(df[df["cluster"] == cluster_id])), random_state=42
    )

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell = GRID_IMG_SIZE
    pad  = 4
    cap  = 22  # caption height in px

    canvas_w = cols * (cell + pad)
    canvas_h = rows * (cell + pad + cap)
    canvas   = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for idx, (_, row) in enumerate(subset.iterrows()):
        col = idx % cols
        r   = idx // cols
        x   = col * (cell + pad)
        y   = r   * (cell + pad + cap)

        # Paste image
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img.thumbnail((cell, cell))
            canvas.paste(img, (x, y))
        except Exception:
            pass  # leave blank if image missing

        # Caption
        draw   = ImageDraw.Draw(canvas)
        title  = row["manga_title"][:20]
        draw.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(30, 30, 30))
        draw.text((x + 3, y + cell + 4), title, fill=(220, 220, 220), font=font)

    return canvas
 

def main():
    results = pd.read_csv(RESULTS_PATH)
    print(f"Loaded results: {len(results)} rows, {results['cluster'].nunique()} clusters")
 
    fig = make_scatter(results)
    fig.write_html(HTML_OUT)
    print(f"Interactive map saved → {HTML_OUT}")
    print("  Open this file in any browser to explore the clusters.")

    SAMPLES_DIR.mkdir(exist_ok=True)
    cluster_ids = sorted(results["cluster"].unique())
 
    for cid in cluster_ids:
        label   = "noise" if cid == -1 else f"cluster_{cid:02d}"
        count   = len(results[results["cluster"] == cid])
        titles  = results[results["cluster"] == cid]["manga_title"].value_counts()
        top3    = ", ".join(titles.head(3).index.tolist())
        print(f"  {label}: {count} pages — top titles: {top3}")
 
        grid = make_cluster_grid(results, cid, n=SAMPLES_PER_CLUSTER)
        grid_path = SAMPLES_DIR / f"{label}.jpg"
        grid.save(grid_path, quality=90)
 
    print(f"\nSample grids saved → {SAMPLES_DIR}/")
    print("\nDone! Next steps:")
    print("  1. Open outputs/cluster_map.html in your browser")
    print("  2. Browse outputs/cluster_samples/ to validate clusters visually")
    print("  3. Tweak KMEANS_K in 3_cluster.py if clusters don't look meaningful")
 
 
if __name__ == "__main__":
    main()