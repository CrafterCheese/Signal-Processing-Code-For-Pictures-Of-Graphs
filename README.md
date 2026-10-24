# axis-calibrated spectral analysis — revised version
This version addresses two issues in the earlier version: subplot labels could sometimes overlap, and the amplitude-envelope panel was disabled whenever the digitized trace failed a strict near-complete-coverage test.

## Install
`python -m pip install numpy matplotlib scipy pillow`
### How To Use
1. Save your full graph image beside the script as building2_graph_with_axes.png, or edit IMAGE_FILENAME near the top of building2_axis_calibrated_spectral_analysis_v2.py.
2. Run the script in PyCharm using an interactive Matplotlib window.
3. Click the two inside corners of the plotting rectangle, excluding tick labels and margins.
4. Click two vertical gridlines with known time labels and enter their values in seconds in the same order.
5. Click two horizontal gridlines with known acceleration labels and enter their values in m/s² in the same order.
6. Open axis_digitization_check_v2.png first. The cyan and blue overlays should follow the green and red traces. If they jump to the legend, gridlines, or wrong curves, do not rely on the spectrum yet.
Choose well-separated calibration lines (several major grid intervals apart if possible). The y-axis values may increase or decrease between clicks; enter the labels exactly as printed.
### What Changed?
 - The final report uses explicit subplot spacing so the upper-row x-axis labels do not overlap the lower-row panel titles.
 - All four panels remain visible. When an envelope or modulation spectrum cannot be estimated, the plot explains why rather than leaving an unexplained blank panel.
 - Color thresholds are slightly more permissive for pale red/green trace pixels.
 - FFT interpolation allows only short pixel gaps and still requires high trace coverage. Otherwise, the trace spectrum uses normalized Lomb–Scargle power from observed points, without filling long gaps.
 - The Hilbert envelope is calculated separately on valid contiguous segments. Only short gaps (up to MAX_ENVELOPE_GAP_PIXELS) are bridged; longer missing intervals remain missing, and edge samples from each segment are discarded to reduce Hilbert boundary artefacts.
- The envelope-modulation spectrum uses normalized Lomb–Scargle power on valid envelope samples. It does not invent data across long gaps.
## Outputs
#### - InsertName_axis_calibrated_spectral_analysis_v2.png: time waveform, spectrum, amplitude envelope and envelope-modulation spectrum or diagnostics.
#### - InsertName_axis_digitization_check_v2.png: the original image with detected trace paths overlaid.
#### - InsertName_axis_calibrated_points_v2.csv: digitized values in seconds and m/s²; missing colored pixels remain blank.
#### - InsertName_axis_calibrated_spectra_v2.csv: spectrum values.
## Scientific limitations
This is still digitization of a raster graph, not the original accelerometer sample array. Pixel thickness, antialiasing, curve overlap, tracking errors and manual calibration contribute uncertainty. Short linear interpolation repairs only small image-detection dropouts; it is not a reconstruction of unmeasured physical data. Segment-wise Hilbert envelopes can still be affected by trace tracking errors. A peak near 0.30 Hz would be evidence of that modulation only if it appears in the analyzed envelope spectrum; it is not inserted by the code. A frequency peak alone does not prove a structural mode or nonlinear coupling. For quantitative validation, analyze the raw accelerometer CSV whenever available.
