# Coin STL Dropper — Streamlit edition

A small browser app that converts children's two-zone coin drawings into watertight STL press discs.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload every file in this folder, including the `.streamlit` folder.
3. Go to `https://share.streamlit.io` and sign in with GitHub.
4. Choose **Create app** → **Yup, I have an app**.
5. Select your repository, branch `main`, and entrypoint `streamlit_app.py`.
6. Deploy.

## Conversion rules

- Drawing inside inner guide circle → recessed/concave on PLA → raised on clay.
- Drawing in outer ring → raised/convex on PLA → indented on clay.

## Default geometry

- 48 mm disc diameter
- 3 mm disc thickness
- 35 mm inner design diameter
- 1.2 mm minimum printable line width
- 0.8 mm centre recess
- 0.8 mm outer-ring raise

The app processes all uploaded files in one batch; there is no fixed class-size count.
