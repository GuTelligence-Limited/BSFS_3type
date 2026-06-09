---
title: BSFS 3-Class Type7 Risk MVP
emoji: B
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
---

# BSFS 3-Class + Type 7 Risk MVP

This Hugging Face Space is an MVP inference demo for the GuTelligence BSFS computer vision model.

The model returns:

- Product primary group:
  - `Type 1/2 hard`
  - `Type 3/4 normal-range`
  - `Type 5/6/7 loose-watery`
- Primary group confidence
- Continuous `Type 7` probability as a risk signal
- Raw 7-class probabilities and top-3 labels

This demo is for product and engineering validation only. It is not a medical diagnostic device.
