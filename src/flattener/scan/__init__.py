"""Single-page document scanner built on ``flattener.dewarp``.

It covers page location, orientation, dewarp and output files.

Modules:

- ``shippable``: the ship gate for model weights.
- ``placement``: where each model view lies in the source, and quarter turns of result grids.
- ``views``: model inputs of placed views, and one dewarp pass placed in the source.
- ``locate``: the page gate's decision, the consensus rule with the pass-0 outline and the
  page conditions (the detect stage).
- ``orient``: OrientNet-1, the page quarter-turn classifier (``orient_train`` trains it).
- ``records``: reason codes, outcomes, scan plans and scan results.
- ``timing``: per-stage timing.
- ``output_policy``: output size and aspect.
- ``io``: reading sources and writing scan files.
- ``fast_check``: the renderer's decisions for a grid's dense map from the grid, a 2 MP
  proxy render and full-resolution sampling (opt-in).
- ``pipeline``: ``Scanner`` (load, plan, render, scan), the candidate chain and fallbacks.
- ``__main__``: ``python -m flattener.scan IMG... --out DIR``.
"""
