"""dcalib - CZT depth-of-interaction photopeak calibration from the adc2kev cache.

Builds one-anode/one-cathode events from the adc2kev calibration cache
(``*.cache.h5``), measures each anode's photopeak position as a function of the
cathode-to-anode ratio r = C/A (a depth proxy), and writes a per-anode ``.dcc``
correction in the legacy Dcalib layout plus a ``depth_summary.csv``. A
PyQt6/pyqtgraph GUI (``dcalib-gui``) inspects, re-fits and reviews the results.
Built on the adc2kev library (cache, calibrations and geometry).

See ``docs/planning/DEPTH_CALIBRATION_PLAN.md`` for the design.
"""

__version__ = "0.1.0"
