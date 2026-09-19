# Coin STL Dropper — Streamlit v4.1 (Direct Sand)

Batch-converts photographed/scanned pupil coin drawings into **3D-printable reeded PLA coin masters** for the Cast the Past direct-sand workflow.

## Canonical v4 workflow

The clay intermediary has been removed.

1. Put the printed PLA coin master on a desk or individual plastic tray.
2. Place a **stainless catering ring** around it.
3. Fill with moulding sand and tamp firmly.
4. Flip the ring/tray assembly over.
5. Remove the PLA master from the underside.
6. Slide the catering ring containing the sand mould onto the larger heatproof casting tray.
7. Adult presenter pours molten low-melt alloy / pewter.
8. Cool, release and compare with the original design.

## v4 STL rules

- round coin / medallion only
- standard **simple reeded vertical edge on every master**
- no meander and no edge-style selector
- no printed outer hoop/collar — the real day-of collar is the stainless catering ring
- vertical sides
- centre-zone drawing is **raised on PLA** and therefore raised on the finished metal coin
- outer design-band drawing is **recessed on PLA** and therefore recessed on the finished metal coin
- optional blind reverse extraction socket for sprung tweezers
- mirroring is **off by default** for the normal flip/remove/direct-sand sequence

## Template

The supplied A4 template uses two blue guide circles and black pupil drawing.

- **Inside the inner circle** → raised detail on the PLA / finished metal coin
- **Between the circles** → recessed detail on the PLA / finished metal coin

Use a bold black pen. Filled black shapes are supported.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

For the existing CastThePast/coin-stl-dropper repository, keep the app in `CoinSTL_Streamlit/` and use `CoinSTL_Streamlit/streamlit_app.py` as the Streamlit entrypoint. Replace files inside that existing folder, not at the repository root. The current package keeps the same dependencies as v3.

## Physical test note

The rear extraction socket is a prototype feature. Default: 5 mm diameter × 1.4 mm deep. Test it with the actual sprung tweezers and adjust only if required. It is a blind hole and must not break through to the design face.

## v4.1 guide detection fix

The app first fits the blue printed guide ellipses, excluding dark artwork from guide detection. Historical template proportions are measured from the photograph and mapped to the selected output dimensions. Monochrome sheets retain the earlier fallback. Reeding and rear extraction socket geometry are unchanged. Tested against the supplied Tom test photo and different synthetic template ratios.

Guide cleanup now follows the detected circle geometry instead of deleting blue/purple pixels throughout the drawing. This preserves coloured underdrawing and tinted dark strokes away from the guides.
