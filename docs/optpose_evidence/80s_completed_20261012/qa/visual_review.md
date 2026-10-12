# Figure review

Question: Does the completed 80S cohort recover low viewing-direction error, and how does its measured trajectory compare with random rotations and the fixed-density control?

Quantitative grid, Curve Endpoint layout, Python/Matplotlib, 7.2 × 3.2 in, PNG 600 dpi, editable PDF/SVG. The left panel shows all 30 measured passes of each defined condition. The right panel shows the last observed endpoint. Fixed cohort: completed DRGN-AI seeds 0,1,2 at every pass; one deterministic density-prior trajectory; six Haar draws. Before/after rows: 126 per-seed rows, 61 aggregated rows per metric. A DRGN-AI-only detail selects the same 30 aggregate rows by method label. All 100,000 defined particles remain in every observation.

All four complete figures and both panels of each were visually inspected. Zero starts and shared vertical ranges retained; endpoint annotations clear the sample-SD bars; shared legends, labels and open axes remain legible. Three-method random baselines compress the low-error traces; the companion detail figures expose their full trajectory. The single deterministic control has zero across-run SD by definition. Every final PDF has a 6 pt minimum glyph size, 0 collision failures and 0 warnings, and passing equal-panel alignment.

Static source validator reports export/alignment items in the renderer because the shared `cryodyna.optpose.plotting.save_figure` owns those operations. The actual helper calls the alignment gate and exports PNG at 600 dpi plus SVG/PDF; all exported files were separately audited. This is a source-scanner scope limitation, recorded alongside its raw report.

Independent endpoint check uses atan2(cross-norm, dot) for viewing angles and traces of relative matrices for full SO(3), conditional on the saved global frame/hand. Largest mean discrepancy is 9.41e-12 degrees. All 60 mean/SD curve aggregates match independent NumPy arithmetic.
