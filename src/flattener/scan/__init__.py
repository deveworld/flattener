"""Single-page document scanner built on ``flattener.dewarp`` (milestone 1).

It covers page location, orientation, dewarp and output files.

Modules:

- ``shippable``: the ship gate for model weights.
- ``placement``: where each model view lies in the source, and quarter turns of result grids.
- ``orient``: OrientNet-1, the page quarter-turn classifier (``orient_train`` trains it).
- ``records``: reason codes, outcomes, scan plans and scan results.
- ``timing``: per-stage timing.
- ``output_policy``: output size and aspect.
- ``io``: reading sources and writing scan files.
"""
