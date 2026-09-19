from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from coin_stl_core import CoinSettings, process_batch


st.set_page_config(
    page_title="Coin STL Dropper",
    page_icon="🪙",
    layout="centered",
)

ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"]


def safe_name(name: str) -> str:
    """Keep a human-readable filename while preventing path traversal / odd characters."""
    base = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(base).stem).strip(" ._") or "drawing"
    suffix = Path(base).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}:
        suffix = ".png"
    return f"{stem}{suffix}"


def numbered_name(index: int, original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}:
        suffix = ".png"
    return f"drawing_{index:03d}{suffix}"


def make_zip_bytes(output_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)
    return buffer.getvalue()


def reset_results() -> None:
    for key in ("result_zip", "result_summary", "result_checks"):
        st.session_state.pop(key, None)


st.title("🪙 Coin STL Dropper")
st.caption("Children's drawings → reeded PLA coin masters → direct sand moulds → cast coins")
st.caption("Engine v4.1 • blue-guide detection • no clay stage • maximum-quality STL generation by default")

st.markdown(
    """
Upload a whole class at once. The two blue guide circles define two **face-design zones**:

- **Inside the inner circle:** raised on the PLA master → **raised on the finished metal coin**
- **In the band between the circles:** recessed on the PLA master → **recessed on the finished metal coin**

Every STL now gets the same **simple reeded vertical edge automatically**. There is no edge-style selector and no clay intermediary.
"""
)

st.info(
    "Direct-sand workflow: PLA coin on tray → stainless catering ring around it → add sand → tamp → flip → "
    "remove PLA master → slide ring/mould to the casting tray → adult pour."
)

with open(Path(__file__).with_name("coin_template_A4.svg"), "rb") as f:
    st.download_button(
        "Download the A4 drawing template",
        data=f.read(),
        file_name="coin_template_A4.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

st.divider()

uploaded_files = st.file_uploader(
    "Drop the class images here",
    type=ALLOWED_TYPES,
    accept_multiple_files=True,
    help="PNG/JPEG are preferred. PDF scans also work. There is no fixed class-size limit.",
)

anonymise = st.checkbox(
    "Use numbered filenames while processing",
    value=False,
    help="Turn this on if filenames contain children's names. Outputs will be drawing_001.stl, drawing_002.stl, etc.",
)

with st.expander("Advanced print settings"):
    c1, c2 = st.columns(2)
    with c1:
        diameter = st.number_input("Coin diameter (mm)", 30.0, 80.0, 48.0, 1.0)
        thickness = st.number_input("Master thickness (mm)", 1.5, 8.0, 3.0, 0.1)
        inner = st.number_input("Inner design diameter (mm)", 20.0, 60.0, 35.0, 0.5)
        line_width = st.number_input("Minimum line width (mm)", 0.5, 3.0, 1.2, 0.1)
    with c2:
        centre_raise = st.number_input("Centre raise height (mm)", 0.2, 2.0, 0.8, 0.1)
        outer_engrave = st.number_input("Outer-band recess depth (mm)", 0.2, 2.0, 0.8, 0.1)
        extraction_hole = st.checkbox(
            "Include reverse extraction socket",
            value=True,
            help="Prototype blind hole for pinched sprung tweezers. It does not pass through the coin face.",
        )
        hole_diameter = st.number_input("Extraction socket diameter (mm)", 2.0, 10.0, 5.0, 0.5, disabled=not extraction_hole)
        hole_depth = st.number_input("Extraction socket depth (mm)", 0.5, 3.0, 1.4, 0.1, disabled=not extraction_hole)

    mirror_master = st.checkbox(
        "Mirror the PLA master left/right (normally OFF)",
        value=False,
        help=(
            "The normal direct-sand flip/remove/cast workflow should preserve the child's orientation, so leave this off. "
            "Use only if a physical test shows your particular handling sequence needs mirroring."
        ),
    )

    quality = st.selectbox(
        "STL quality",
        ["Maximum quality (default)", "High / faster", "Standard / smaller files"],
        index=0,
        help=(
            "Maximum quality uses a 2048 px working image and a dense 192 × 768 mesh. "
            "This is also dense enough to reproduce the standard reeded edge cleanly."
        ),
    )
    st.caption("Simple reeding is applied automatically to every coin edge; there is no edge-style choice.")

if uploaded_files:
    st.info(f"{len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''} ready to process.")

if st.button(
    "MAKE REEDED COIN STL FILES",
    type="primary",
    use_container_width=True,
    disabled=not uploaded_files,
):
    reset_results()
    if quality == "Standard / smaller files":
        output_pixels, radial_rings, angular_segments = 1024, 96, 384
    elif quality == "High / faster":
        output_pixels, radial_rings, angular_segments = 1536, 128, 576
    else:
        output_pixels, radial_rings, angular_segments = 2048, 192, 768

    settings = CoinSettings(
        disc_diameter_mm=float(diameter),
        disc_thickness_mm=float(thickness),
        inner_design_diameter_mm=float(inner),
        min_line_width_mm=float(line_width),
        centre_raise_height_mm=float(centre_raise),
        outer_engrave_depth_mm=float(outer_engrave),
        include_extraction_hole=bool(extraction_hole),
        extraction_hole_diameter_mm=float(hole_diameter),
        extraction_hole_depth_mm=float(hole_depth),
        output_pixels=output_pixels,
        mesh_radial_rings=radial_rings,
        mesh_angular_segments=angular_segments,
        mirror_master=bool(mirror_master),
    )

    try:
        settings.validate()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    progress_bar = st.progress(0.0, text="Preparing files…")
    status_box = st.empty()

    with tempfile.TemporaryDirectory(prefix="coin_stl_") as temp_root:
        root = Path(temp_root)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_paths = []
        used_names: set[str] = set()
        for i, uploaded in enumerate(uploaded_files, start=1):
            chosen = numbered_name(i, uploaded.name) if anonymise else safe_name(uploaded.name)
            candidate = chosen
            stem, suffix = Path(chosen).stem, Path(chosen).suffix
            n = 2
            while candidate.lower() in used_names:
                candidate = f"{stem}__{n}{suffix}"
                n += 1
            used_names.add(candidate.lower())
            path = input_dir / candidate
            path.write_bytes(uploaded.getvalue())
            input_paths.append(path)

        total = len(input_paths)

        def progress(message: str) -> None:
            status_box.caption(message)
            if message.startswith("[") and "/" in message:
                try:
                    current_index = int(message.split("/", 1)[0].lstrip("[")) - 1
                    progress_bar.progress(max(0.0, min(1.0, current_index / total)), text=message)
                except Exception:
                    pass

        results = process_batch(input_paths, output_dir, settings, progress=progress)
        progress_bar.progress(1.0, text="Finished")

        checks = []
        for result in results:
            if result.success and result.diagnostic_file:
                diag = Path(result.diagnostic_file)
                if diag.exists():
                    checks.append((diag.name, diag.read_bytes()))

        ok = sum(1 for r in results if r.success)
        failures = [(Path(r.input_file).name, r.error) for r in results if not r.success]
        st.session_state.result_zip = make_zip_bytes(output_dir)
        st.session_state.result_summary = (ok, len(results), failures)
        st.session_state.result_checks = checks

if "result_zip" in st.session_state:
    ok, total, failures = st.session_state.result_summary
    if ok == total:
        st.success(f"Done — {ok}/{total} reeded coin STL files created successfully.")
    else:
        st.warning(f"Created {ok}/{total} STL files. {total - ok} file(s) need attention.")

    st.download_button(
        "DOWNLOAD ALL STLs + CHECK IMAGES (.ZIP)",
        data=st.session_state.result_zip,
        file_name="Coin_STL_Output_v4_Direct_Sand.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    if failures:
        with st.expander("Files that did not convert"):
            for name, error in failures:
                st.write(f"**{name}** — {error}")

    checks = st.session_state.get("result_checks", [])
    if checks:
        with st.expander("Quick visual check of interpreted drawings"):
            st.caption(
                "Blue = centre-zone drawing (raised on PLA / raised on metal). "
                "Green = outer design band (recessed on PLA / recessed on metal). "
                "The reeded side edge is added automatically and is not shown in this top-view check image."
            )
            for name, image_bytes in checks:
                st.image(image_bytes, caption=name, use_container_width=True)

st.divider()
st.caption(
    "Privacy note: uploaded files are sent to the Streamlit server for processing. "
    "This app deletes its temporary processing files after building the ZIP, but you should still avoid "
    "including pupils' names, faces or other personal data unless your school's policy permits cloud processing."
)
