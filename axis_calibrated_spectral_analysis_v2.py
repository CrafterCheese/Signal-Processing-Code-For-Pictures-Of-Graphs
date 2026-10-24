"""Axis-calibrated spectral analysis from a plotted accelerometer image.

This script digitizes the colored X/Y traces from a graph image that includes
its labelled axes and grid. It converts image pixels to time (s) and
acceleration (m/s^2) using four manually calibrated tick points, then computes
spectra from the extracted trace. It does NOT synthesize frequencies.

IMPORTANT
---------
A raster graph is still a digitized approximation of the original sensor data.
For publication-grade measured results, use the original time-series CSV.
The manual calibration is intentional: it avoids just trusting OCR or
assuming axis limits/tick spacing from an image.

Dependencies: numpy, matplotlib, scipy, pillow
Install: python -m pip install numpy matplotlib scipy pillow

Workflow
--------
1. Save the graph as IMAGE_FILENAME.
2. Run this file.
3. Click top-left and bottom-right corners of the plotting rectangle (inside
   the axes frame; exclude tick labels and axis titles).
4. Click two vertical grid lines with known time labels, then enter those times.
5. Click two horizontal grid lines with known acceleration labels, then enter
   those acceleration values in m/s^2.
6. Inspect the trace overlay and spectrum; exported CSVs contain calibrated
   digitized points and spectrum values.
"""
from __future__ import annotations

import csv
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

# USER SETTINGS, IMPORTANT DO NOT SKIP
IMAGE_FILENAME = "InsertNameHere" #insert the file path of the image (or if in same folder just the file name)
OUTPUT_FIGURE = "InsertName_axis_calibrated_spectral_analysis_v2.png"
OUTPUT_OVERLAY = "InsertName_axis_digitization_check_v2.png"
OUTPUT_TRACE_CSV = "InsertName_axis_calibrated_points_v2.csv"
OUTPUT_SPECTRA_CSV = "InsertName_axis_calibrated_spectra_v2.csv"

MAX_FREQUENCY_HZ = 6.0
MAX_ENVELOPE_FREQUENCY_HZ = 1.5
MIN_FREQUENCY_HZ = 0.10
MAX_PEAK_LABELS = 4

# Colour detection tuned for the original muted green/red trace colours.
# Lower MIN_SATURATION slightly if traces are pale after screenshot scaling.
MIN_SATURATION = 6
MIN_BRIGHTNESS = 45
GREEN_HUE_RANGE = (30, 135)  # PIL HSV hue scale: 0..255
RED_HUE_LOW_MAX = 22
RED_HUE_HIGH_MIN = 238

# Short raster-detection gaps can be interpolated for a uniform FFT. Larger
# gaps are not filled; the spectrum then uses Lomb–Scargle on observed points.
MAX_SHORT_GAP_PIXELS = 8
MIN_COVERAGE_FOR_FFT = 0.92
# The envelope is estimated independently on valid contiguous segments. Only
# likely raster/colour-detection gaps up to this limit are interpolated; longer
# gaps remain missing and split the signal into separate segments.
MAX_ENVELOPE_GAP_PIXELS = 10
MIN_ENVELOPE_SEGMENT_SECONDS = 1.75
ENVELOPE_EDGE_TRIM_SECONDS = 0.15
MIN_ENVELOPE_COVERAGE_FOR_SPECTRUM = 0.50

# Original figure palette.
C_GREEN = "#6aa84f"
C_RED = "#e06666"
C_BLACK = "#111111"
C_DARK_GREEN = "#38761d"
GRID_COLOR = "#b7b7b7"


def load_rgb(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}\n"
            f"Save your graph image as '{IMAGE_FILENAME}' beside this script, "
            "or change IMAGE_FILENAME near the top of the script."
        )
    im = Image.open(image_path).convert("RGB")
    rgb = np.asarray(im)
    hsv = np.asarray(im.convert("HSV"))
    return rgb, hsv


def choose_points(rgb: np.ndarray, n: int, title: str, instruction: str,
                  prior_points: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """Display image and use mouse clicks to select pixel coordinates."""
    height, width = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(rgb, extent=(-0.5, width - 0.5, height - 0.5, -0.5), interpolation="nearest")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_title(title + "\n" + instruction, fontsize=10)
    ax.set_xlabel("Image x coordinate (pixels)")
    ax.set_ylabel("Image y coordinate (pixels; increases downward)")
    if prior_points:
        for i, (px, py) in enumerate(prior_points, start=1):
            ax.plot(px, py, marker="o", ms=6, mfc="none", mec="#0066cc", mew=1.5)
            ax.text(px + 5, py - 5, str(i), color="#0066cc", weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    fig.tight_layout()
    fig.canvas.draw()
    print(f"\n{title}\n{instruction}")
    print(f"Click {n} point(s) in the image window, in the stated order. Close only after selecting all points.")
    points = plt.ginput(n=n, timeout=-1, show_clicks=True)
    plt.close(fig)
    if len(points) != n:
        raise RuntimeError(
            f"Expected {n} clicks but received {len(points)}. Re-run the script and complete each calibration step."
        )
    return [(float(x), float(y)) for x, y in points]


def ask_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            value = float(raw)
            if np.isfinite(value):
                return value
        except ValueError:
            pass
        print("Please enter a numeric value, e.g. 18 or -5.0.")


def axis_calibration(rgb: np.ndarray) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return plot bbox (left, top, right, bottom), x calibration and y calibration."""
    h, w = rgb.shape[:2]
    box_pts = choose_points(
        rgb, 2, "Step 1/3 — Select the plotting rectangle",
        "Click 1: top-left inside the plot frame. Click 2: bottom-right inside the plot frame.\n"
        "Exclude tick labels, tick numbers, axis titles, and margins. Include all plotted traces."
    )
    (bx1, by1), (bx2, by2) = box_pts
    left, right = sorted((bx1, bx2))
    top, bottom = sorted((by1, by2))
    left = max(0.0, left); right = min(float(w - 1), right)
    top = max(0.0, top); bottom = min(float(h - 1), bottom)
    if right - left < 30 or bottom - top < 30:
        raise ValueError("Selected plotting rectangle is too small. Re-run and click the plot's inside corners.")

    x_pts = choose_points(
        rgb, 2, "Step 2/3 — Calibrate time (x-axis)",
        "Click two vertical grid lines/ticks with clearly readable time labels.\n"
        "Click them in the order you will enter their time values (seconds)."
    )
    x1, x2 = x_pts[0][0], x_pts[1][0]
    t1 = ask_float("Time printed at vertical gridline #1 (seconds): ")
    t2 = ask_float("Time printed at vertical gridline #2 (seconds): ")
    if abs(x2 - x1) < 5 or abs(t2 - t1) < 1e-12:
        raise ValueError("The two time calibration points must be distinct and separated on the x-axis.")
    seconds_per_pixel = (t2 - t1) / (x2 - x1)

    y_pts = choose_points(
        rgb, 2, "Step 3/3 — Calibrate acceleration (y-axis)",
        "Click two horizontal grid lines/ticks with clearly readable acceleration labels.\n"
        "Click them in the order you will enter their values (m/s^2)."
    )
    y1, y2 = y_pts[0][1], y_pts[1][1]
    a1 = ask_float("Acceleration printed at horizontal gridline #1 (m/s^2): ")
    a2 = ask_float("Acceleration printed at horizontal gridline #2 (m/s^2): ")
    if abs(y2 - y1) < 5 or abs(a2 - a1) < 1e-12:
        raise ValueError("The two acceleration calibration points must be distinct and separated on the y-axis.")
    accel_per_pixel = (a2 - a1) / (y2 - y1)

    print("\nCalibration captured:")
    print(f"  time = {t1:g} s at x={x1:.1f}px; {t2:g} s at x={x2:.1f}px")
    print(f"  acceleration = {a1:g} m/s^2 at y={y1:.1f}px; {a2:g} m/s^2 at y={y2:.1f}px")
    print(f"  x scale = {seconds_per_pixel:.7g} s/pixel; y scale = {accel_per_pixel:.7g} (m/s^2)/pixel")
    return (left, top, right, bottom), (x1, t1, x2, t2), (y1, a1, y2, a2)


def colour_mask(hsv: np.ndarray, which: str) -> np.ndarray:
    hue = hsv[..., 0].astype(int)
    sat = hsv[..., 1].astype(int)
    val = hsv[..., 2].astype(int)
    vivid = (sat >= MIN_SATURATION) & (val >= MIN_BRIGHTNESS)
    if which == "green":
        return vivid & (hue >= GREEN_HUE_RANGE[0]) & (hue <= GREEN_HUE_RANGE[1])
    if which == "red":
        return vivid & ((hue <= RED_HUE_LOW_MAX) | (hue >= RED_HUE_HIGH_MIN))
    raise ValueError("which must be green or red")


def _clusters(y_indices: np.ndarray, max_vertical_gap: int = 2) -> list[np.ndarray]:
    if y_indices.size == 0:
        return []
    split_at = np.flatnonzero(np.diff(y_indices) > max_vertical_gap) + 1
    return list(np.split(y_indices, split_at))


def track_trace_rows(hsv_plot: np.ndarray, which: str) -> np.ndarray:
    """Pick a colored trace row per image column using colour + continuity.

    It returns NaN where the colour was not observed. It does not synthesize
    missing samples; interpolation is only considered later for short raster gaps.
    """
    mask = colour_mask(hsv_plot, which)
    rows = np.full(mask.shape[1], np.nan, dtype=float)
    prev = None
    missing_run = 0
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        groups = _clusters(ys)
        if not groups:
            missing_run += 1
            if missing_run > 8:
                prev = None
            continue
        centers = np.array([float(np.median(g)) for g in groups])
        sizes = np.array([len(g) for g in groups])
        if prev is None:
            # First point or after a long gap: pick most substantial coloured cluster.
            j = int(np.argmax(sizes))
        else:
            # Nearest cluster maintains the continuous waveform across columns.
            # A tiny preference for thicker clusters breaks near-ties.
            distances = np.abs(centers - prev) - 0.15 * sizes
            j = int(np.argmin(distances))
        rows[x] = centers[j]
        prev = centers[j]
        missing_run = 0
    return rows


def map_pixels_to_values(x_pixels: np.ndarray, y_rows: np.ndarray,
                         x_cal: tuple[float, float, float, float],
                         y_cal: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x1, t1, x2, t2 = x_cal
    y1, a1, y2, a2 = y_cal
    time_s = t1 + (x_pixels - x1) * (t2 - t1) / (x2 - x1)
    acceleration = a1 + (y_rows - y1) * (a2 - a1) / (y2 - y1)
    acceleration[~np.isfinite(y_rows)] = np.nan
    return time_s, acceleration


def fill_only_short_gaps(y: np.ndarray, max_gap: int) -> tuple[np.ndarray, int, int]:
    out = y.copy()
    missing = ~np.isfinite(out)
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    filled = 0
    largest = 0
    for start, end in zip(starts, ends):
        gap = int(end - start)
        largest = max(largest, gap)
        if gap <= max_gap and start > 0 and end < len(out):
            out[start:end] = np.interp(np.arange(start, end), [start - 1, end], [out[start - 1], out[end]])
            filled += gap
    return out, filled, largest


def detrend_observed(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(t) & np.isfinite(y)
    tt, yy = t[ok], y[ok]
    if len(tt) < 32:
        raise ValueError("Fewer than 32 coloured trace points were detected. Adjust colour thresholds or use a clearer image.")
    return tt, signal.detrend(yy, type="linear")


def lomb_spectrum(t: np.ndarray, y: np.ndarray, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalized Lomb–Scargle spectrum from actual detected image points only."""
    tt, yy = detrend_observed(t, y)
    duration = float(tt[-1] - tt[0])
    nominal_resolution = 1.0 / duration
    df = nominal_resolution / 8.0
    freq = np.arange(max(fmin, nominal_resolution), fmax, df)
    if len(freq) < 3:
        raise ValueError("The plotted time span is too short for this frequency range.")
    power = signal.lombscargle(tt - tt[0], yy, 2.0 * np.pi * freq,
                               normalize=True, floating_mean=True)
    peak = float(np.max(power)) if len(power) else 0.0
    if peak > 0:
        power = power / peak
    return freq, power, nominal_resolution


def fft_amplitude_spectrum(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided Hann-window amplitude spectrum, units inherited from y."""
    if not np.all(np.isfinite(y)):
        raise ValueError("FFT input still has missing values.")
    dt = float(np.median(np.diff(t)))
    yy = signal.detrend(y, type="linear")
    win = np.hanning(len(yy))
    nfft = max(8192, 2 ** int(np.ceil(np.log2(max(2, len(yy) * 8)))))
    spec = np.abs(np.fft.rfft(yy * win, n=nfft)) * (2.0 / np.sum(win))
    if len(spec) > 0:
        spec[0] *= 0.5
    if nfft % 2 == 0 and len(spec) > 1:
        spec[-1] *= 0.5
    freq = np.fft.rfftfreq(nfft, dt)
    return freq, spec


def list_peaks(freq: np.ndarray, vals: np.ndarray, max_peaks: int = MAX_PEAK_LABELS,
               min_frequency: float = MIN_FREQUENCY_HZ) -> list[tuple[float, float]]:
    in_band = (freq >= min_frequency) & (freq <= MAX_FREQUENCY_HZ)
    f, v = freq[in_band], vals[in_band]
    if len(f) < 3 or not np.any(np.isfinite(v)) or np.nanmax(v) <= 0:
        return []
    df = float(np.median(np.diff(f)))
    min_dist = max(1, int(round(0.12 / df)))
    prom = max(0.03 * float(np.nanmax(v)), np.finfo(float).eps)
    pks, _ = signal.find_peaks(v, prominence=prom, distance=min_dist)
    if not len(pks):
        pks = np.array([int(np.nanargmax(v))])
    ranked = sorted(pks, key=lambda i: v[i], reverse=True)[:max_peaks]
    return [(float(f[i]), float(v[i])) for i in ranked]


def trace_is_fft_ready(y: np.ndarray) -> tuple[bool, np.ndarray, int, int, float]:
    coverage = float(np.mean(np.isfinite(y)))
    filled, filled_count, largest_gap = fill_only_short_gaps(y, MAX_SHORT_GAP_PIXELS)
    ready = (coverage >= MIN_COVERAGE_FOR_FFT) and np.all(np.isfinite(filled))
    return ready, filled, filled_count, largest_gap, coverage


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index ranges where mask is True."""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def gap_aware_hilbert_envelope(
    t: np.ndarray,
    y: np.ndarray,
    max_gap_pixels: int = MAX_ENVELOPE_GAP_PIXELS,
    min_segment_seconds: float = MIN_ENVELOPE_SEGMENT_SECONDS,
    edge_trim_seconds: float = ENVELOPE_EDGE_TRIM_SECONDS,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Estimate Hilbert envelope on valid contiguous segments only.

    Only gaps no longer than max_gap_pixels are linearly bridged, since these
    are plausibly raster/colour-detection dropouts. Longer gaps remain NaN and
    split the trace into segments. Each segment is processed independently;
    edge samples are discarded to reduce Hilbert boundary artefacts.

    Returns (filled_carrier, envelope_with_NaNs, count_filled, largest_gap,
    fraction_of_samples_with_a_valid_envelope).
    """
    if len(t) != len(y) or len(y) < 32:
        raise ValueError("Envelope input must have matching time/signal arrays with at least 32 samples.")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time samples must be strictly increasing to compute an envelope.")

    carrier, filled_count, largest_gap = fill_only_short_gaps(y, max_gap_pixels)
    envelope = np.full(carrier.shape, np.nan, dtype=float)
    valid = np.isfinite(carrier)
    min_samples = max(32, int(np.ceil(min_segment_seconds / dt)) + 1)
    trim = max(2, int(np.ceil(edge_trim_seconds / dt)))

    for start, end in contiguous_true_runs(valid):
        segment = carrier[start:end]
        if len(segment) < min_samples or len(segment) <= 2 * trim + 8:
            continue
        segment_detrended = signal.detrend(segment, type="linear")
        segment_env = np.abs(signal.hilbert(segment_detrended))
        # Preserve only the segment interior where the finite-record Hilbert
        # transform is less affected by end effects.
        envelope[start + trim:end - trim] = segment_env[trim:-trim]

    envelope_coverage = float(np.mean(np.isfinite(envelope)))
    return carrier, envelope, filled_count, largest_gap, envelope_coverage


def envelope_modulation_spectrum(
    t: np.ndarray, envelope: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[float, float]], float | None]:
    """Use Lomb–Scargle on valid envelope samples, never fill long gaps."""
    ok = np.isfinite(t) & np.isfinite(envelope)
    if int(np.count_nonzero(ok)) < 64:
        return None, None, [], None
    tv, ev = t[ok], envelope[ok]
    span = float(tv[-1] - tv[0])
    if span < 5.0 or np.mean(ok) < MIN_ENVELOPE_COVERAGE_FOR_SPECTRUM:
        return None, None, [], None
    fmin = max(0.05, 1.0 / span)
    if fmin >= MAX_ENVELOPE_FREQUENCY_HZ:
        return None, None, [], None
    f, power, resolution = lomb_spectrum(tv, ev, fmin, MAX_ENVELOPE_FREQUENCY_HZ)
    peaks = list_peaks(f, power, max_peaks=3, min_frequency=fmin)
    return f, power, peaks, resolution


def save_csvs(t: np.ndarray, ay: np.ndarray, ax: np.ndarray, y_rows: np.ndarray,
              x_rows: np.ndarray, spec_freq: np.ndarray, spec_y: np.ndarray,
              spec_x: np.ndarray, method: str) -> None:
    total = np.where(np.isfinite(ay) & np.isfinite(ax), np.sqrt(ay ** 2 + ax ** 2), np.nan)
    with Path(OUTPUT_TRACE_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ay_mps2", "ax_mps2", "total_horizontal_mps2", "y_trace_pixel_row", "x_trace_pixel_row"])
        for row in zip(t, ay, ax, total, y_rows, x_rows):
            writer.writerow(["" if not np.isfinite(z) else f"{z:.8g}" for z in row])
    with Path(OUTPUT_SPECTRA_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", f"y_spectrum_{method}", f"x_spectrum_{method}"])
        for row in zip(spec_freq, spec_y, spec_x):
            writer.writerow([f"{z:.8g}" for z in row])


def main() -> None:
    rgb, hsv = load_rgb(Path(IMAGE_FILENAME))
    h, w = rgb.shape[:2]
    if w < 250 or h < 150:
        warnings.warn("This image is small; pixel digitization may limit frequency accuracy.")

    bbox, x_cal, y_cal = axis_calibration(rgb)
    left, top, right, bottom = bbox
    ix0, ix1 = int(np.ceil(left)), int(np.floor(right))
    iy0, iy1 = int(np.ceil(top)), int(np.floor(bottom))
    rgb_plot = rgb[iy0:iy1 + 1, ix0:ix1 + 1]
    hsv_plot = hsv[iy0:iy1 + 1, ix0:ix1 + 1]
    if rgb_plot.size == 0:
        raise ValueError("Selected plot rectangle produced an empty crop. Re-run and select valid corners.")

    y_local = track_trace_rows(hsv_plot, "green")
    x_local = track_trace_rows(hsv_plot, "red")
    cols = np.arange(ix0, ix1 + 1, dtype=float)
    rows_y = y_local + iy0
    rows_x = x_local + iy0
    t, ay = map_pixels_to_values(cols, rows_y, x_cal, y_cal)
    t_x, ax = map_pixels_to_values(cols, rows_x, x_cal, y_cal)
    if not np.allclose(t, t_x):
        raise RuntimeError("Internal error: time calibration differs between traces.")

    # Keep time-ordered plot columns. Values outside plotted axis domain are naturally excluded by bbox.
    valid_t = np.isfinite(t)
    t, ay, ax = t[valid_t], ay[valid_t], ax[valid_t]
    rows_y, rows_x = rows_y[valid_t], rows_x[valid_t]
    if len(t) < 100:
        raise ValueError("Selected region contains too few columns; select the actual plot area, not the full screenshot.")

    cover_y = float(np.mean(np.isfinite(ay)))
    cover_x = float(np.mean(np.isfinite(ax)))
    if cover_y < 0.50 or cover_x < 0.50:
        raise ValueError(
            f"Trace coverage is low (green/Y={cover_y:.1%}, red/X={cover_x:.1%}). "
            "Try lowering MIN_SATURATION to 7 or adjusting the colour ranges near the top of the script."
        )

    # Spectra: FFT amplitude (m/s^2) only if trace digitization is nearly complete
    # and all gaps can be filled only across a few pixels. Otherwise use Lomb-Scargle
    # normalized power, which leaves missing trace points missing.
    ready_y, ay_uniform, fill_y, max_gap_y, cover_y = trace_is_fft_ready(ay)
    ready_x, ax_uniform, fill_x, max_gap_x, cover_x = trace_is_fft_ready(ax)
    fft_ready = ready_y and ready_x
    if fft_ready:
        f_y_full, s_y_full = fft_amplitude_spectrum(t, ay_uniform)
        f_x_full, s_x_full = fft_amplitude_spectrum(t, ax_uniform)
        freq = f_y_full
        spec_y = s_y_full
        spec_x = np.interp(freq, f_x_full, s_x_full)
        method = "hann_fft_amplitude_mps2"
        spectrum_ylabel = "Spectral amplitude (m/s²)"
        spectrum_title = "B2. Windowed FFT Spectrum (axis-calibrated trace)"
        freq_resolution = 1.0 / (t[-1] - t[0])
        peaks_y = list_peaks(freq, spec_y)
        peaks_x = list_peaks(freq, spec_x)
    else:
        f_y, p_y, resolution_y = lomb_spectrum(t, ay, MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
        f_x, p_x, resolution_x = lomb_spectrum(t, ax, MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
        freq = f_y
        spec_y = p_y
        spec_x = np.interp(freq, f_x, p_x)
        method = "normalized_lomb_scargle_power"
        spectrum_ylabel = "Normalized Lomb–Scargle power"
        spectrum_title = "B2. Lomb–Scargle Spectrum (missing pixels not filled)"
        freq_resolution = max(resolution_y, resolution_x)
        peaks_y = list_peaks(freq, spec_y)
        peaks_x = list_peaks(freq, spec_x)

    # Envelope processing is gap-aware and independent of whether the full FFT is
    # possible. Short pixel-detection gaps are interpolated; longer gaps remain NaN
    # and split the trace into contiguous sections. Each section is analyzed on its
    # own, with edge samples discarded to limit Hilbert boundary artifacts.
    ay_env_carrier, envelope, env_fill, env_max_gap, env_coverage = gap_aware_hilbert_envelope(t, ay)
    f_env, s_env, env_peaks, env_resolution = envelope_modulation_spectrum(t, envelope)

    save_csvs(t, ay, ax, rows_y, rows_x, freq, spec_y, spec_x, method)

    # Save a digitization diagnostic so the user can visually verify the tracked
    # pixel paths against the original coloured traces before trusting the spectrum.
    fig_check, ax_check = plt.subplots(figsize=(13, 7), dpi=150)
    ax_check.imshow(rgb, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="nearest")
    ax_check.plot(cols, rows_y, color="#00e5ff", lw=0.9, alpha=0.95, label="Detected Y trace (overlay)")
    ax_check.plot(cols, rows_x, color="#205cff", lw=0.9, alpha=0.95, label="Detected X trace (overlay)")
    ax_check.set_xlim(ix0 - 5, ix1 + 5)
    ax_check.set_ylim(iy1 + 5, iy0 - 5)
    ax_check.set_title("Digitization Check — overlays should follow the green and red traces")
    ax_check.set_xlabel("Image x coordinate (pixels)")
    ax_check.set_ylabel("Image y coordinate (pixels; increases downward)")
    ax_check.legend(loc="upper right", framealpha=0.9)
    fig_check.tight_layout()
    fig_check.savefig(OUTPUT_OVERLAY, dpi=240, bbox_inches="tight")
    plt.close(fig_check)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 10, "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    })
    # Explicit spacing prevents the previous row's x-axis label from colliding
    # with the next row's title. Keep all four panels visible even when an
    # envelope or its modulation spectrum cannot be estimated reliably.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=170,
                             gridspec_kw={"width_ratios": [1.15, 1], "height_ratios": [1, 1]})
    ax_t, ax_f, ax_env, ax_envf = axes.ravel()
    fig.suptitle("Building 2: Axis-Calibrated Roof Accelerometer & Spectral Analysis",
                 fontsize=14, fontweight="bold", y=0.975)

    total = np.where(np.isfinite(ay) & np.isfinite(ax), np.sqrt(ay ** 2 + ax ** 2), np.nan)
    ax_t.plot(t, ax, color=C_RED, alpha=0.82, lw=1.0, label="S-Wave (Horizontal X — Torsion)")
    ax_t.plot(t, ay, color=C_GREEN, alpha=0.88, lw=1.1, label="S-Wave (Horizontal Y — Shaking Axis)")
    ax_t.plot(t, total, color=C_BLACK, lw=1.1, label="Total horizontal magnitude $|a|$")
    ax_t.axhline(0, color="#999999", lw=0.7, alpha=0.7)
    ax_t.set_title("A2. Digitized Time-Domain Waveform", pad=9)
    ax_t.set_xlabel("Time (s)", labelpad=5)
    ax_t.set_ylabel("Acceleration (m/s²)")
    ax_t.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax_t.legend(loc="upper left", framealpha=0.9)

    view = (freq >= 0) & (freq <= MAX_FREQUENCY_HZ)
    ax_f.plot(freq[view], spec_y[view], color=C_DARK_GREEN, lw=1.8, label="Y-axis spectrum")
    ax_f.fill_between(freq[view], 0, spec_y[view], color=C_GREEN, alpha=0.22)
    ax_f.plot(freq[view], spec_x[view], color="#cc0000", lw=1.25, ls="--", alpha=0.88, label="X-axis spectrum")
    visible_peak_values = [v for _, v in peaks_y[:MAX_PEAK_LABELS] if np.isfinite(v)]
    spectrum_top = float(np.nanmax(spec_y[view])) if np.any(view) else 1.0
    for i, (fp, vp) in enumerate(peaks_y[:MAX_PEAK_LABELS]):
        text_y = min(spectrum_top * 1.06, max(vp * 0.78, spectrum_top * (0.92 - 0.13 * i)))
        text_x = fp + (0.10 if i % 2 == 0 else -0.10)
        ax_f.scatter([fp], [vp], s=24, color=C_DARK_GREEN, edgecolors="white", linewidths=0.4, zorder=5)
        ax_f.annotate(f"{fp:.2f} Hz", xy=(fp, vp), xytext=(text_x, text_y),
                      ha="left" if i % 2 == 0 else "right", fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7))
    ax_f.set_xlim(0, MAX_FREQUENCY_HZ)
    ax_f.set_title(spectrum_title, pad=9)
    ax_f.set_xlabel("Frequency (Hz)", labelpad=5)
    ax_f.set_ylabel(spectrum_ylabel)
    ax_f.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax_f.legend(loc="upper right", framealpha=0.9)

    observed_y = np.isfinite(ay)
    if np.any(observed_y):
        ax_env.plot(t, ay, color=C_GREEN, alpha=0.28, lw=0.8, label="Digitized Y trace (gaps retained)")
    if np.any(np.isfinite(envelope)):
        ax_env.plot(t, envelope, color=C_DARK_GREEN, lw=1.45,
                    label="Hilbert envelope (valid segments only)")
        ax_env.set_title("C2. Y-Axis Amplitude Envelope", pad=9)
        ax_env.legend(loc="upper right", framealpha=0.9)
    else:
        ax_env.text(0.04, 0.92,
                    "No sufficiently long continuous trace segment was available for a reliable envelope.\n"
                    "Try improving trace detection/overlay or use the original sensor CSV.",
                    transform=ax_env.transAxes, va="top", fontsize=9, wrap=True)
        ax_env.set_title("C2. Y-Axis Amplitude Envelope — unavailable", pad=9)
    ax_env.axhline(0, color="#999999", lw=0.7, alpha=0.7)
    ax_env.set_xlabel("Time (s)", labelpad=5)
    ax_env.set_ylabel("Acceleration / envelope (m/s²)")
    ax_env.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)

    if f_env is not None and s_env is not None:
        mask_env = (f_env >= 0.05) & (f_env <= MAX_ENVELOPE_FREQUENCY_HZ)
        ax_envf.plot(f_env[mask_env], s_env[mask_env], color=C_DARK_GREEN, lw=1.7,
                     label="Envelope modulation (normalized Lomb–Scargle power)")
        ax_envf.fill_between(f_env[mask_env], 0, s_env[mask_env], color=C_GREEN, alpha=0.22)
        for i, (fp, vp) in enumerate(env_peaks):
            ax_envf.scatter([fp], [vp], s=24, color=C_GREEN, edgecolors="white", linewidths=0.4, zorder=5)
            ax_envf.annotate(f"{fp:.2f} Hz", xy=(fp, vp),
                             xytext=(fp + (0.04 if i % 2 == 0 else -0.04), min(0.98, vp + 0.08)),
                             fontsize=8, ha="left" if i % 2 == 0 else "right",
                             arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7))
        ax_envf.set_xlim(0, MAX_ENVELOPE_FREQUENCY_HZ)
        ax_envf.set_ylim(0, 1.08)
        ax_envf.set_title("D2. Envelope-Modulation Spectrum", pad=9)
        ax_envf.set_xlabel("Modulation frequency (Hz)", labelpad=5)
        ax_envf.set_ylabel("Normalized Lomb–Scargle power")
        ax_envf.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax_envf.legend(loc="upper right", framealpha=0.9)
    else:
        ax_envf.text(0.04, 0.92,
                     "Envelope modulation spectrum not calculated.\n"
                     f"Valid envelope coverage: {env_coverage:.1%}.\n"
                     "Long missing intervals were excluded, not fabricated.",
                     transform=ax_envf.transAxes, va="top", fontsize=9, wrap=True)
        ax_envf.set_title("D2. Envelope-Modulation Spectrum — unavailable", pad=9)
        ax_envf.set_xlabel("Modulation frequency (Hz)", labelpad=5)
        ax_envf.set_ylabel("Normalized Lomb–Scargle power")
        ax_envf.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)

    duration = float(t[-1] - t[0])
    footer1 = (f"Image-digitized approximation | duration={duration:.3f} s | nominal resolution≈{freq_resolution:.3f} Hz | "
               f"trace coverage: Y={cover_y:.1%}, X={cover_x:.1%} | short gaps filled: FFT Y={fill_y}, X={fill_x}; "
               f"envelope Y={env_fill}; envelope coverage={env_coverage:.1%}")
    caution = ("Spectrum: Hann-window FFT amplitude (m/s²)." if fft_ready else
               "Spectrum: normalized Lomb–Scargle power; missing trace points were not filled for this spectrum.")
    fig.text(0.5, 0.055, footer1, ha="center", va="bottom", fontsize=7.5)
    fig.text(0.5, 0.025,
             caution + " Envelope spectrum is normalized Lomb–Scargle power. Digitization is approximate; raw sensor data is preferred.",
             ha="center", va="bottom", fontsize=7.5, color="#8a3b12", fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.14, hspace=0.43, wspace=0.28)
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")

    print("\nAnalysis complete.")
    print(f"Image: {Path(IMAGE_FILENAME).resolve()}")
    print(f"Figure saved: {Path(OUTPUT_FIGURE).resolve()}")
    print(f"Digitization check saved: {Path(OUTPUT_OVERLAY).resolve()}")
    print(f"Digitized calibrated points CSV: {Path(OUTPUT_TRACE_CSV).resolve()}")
    print(f"Spectrum CSV: {Path(OUTPUT_SPECTRA_CSV).resolve()}")
    print(f"Calibrated time span: {t[0]:.4f} to {t[-1]:.4f} s ({duration:.3f} s)")
    print(f"Nominal frequency resolution: approximately {freq_resolution:.4f} Hz; zero-padding does not improve this limit.")
    print(f"Colour trace coverage: Y={cover_y:.1%}, X={cover_x:.1%}")
    print(f"Envelope coverage after segment-wise Hilbert transform: {env_coverage:.1%}")
    print(f"Envelope short-gap points interpolated: {env_fill}; largest detected gap: {env_max_gap} pixels")
    print(f"Spectrum method: {method}")
    print("Prominent Y-axis frequency candidates (not automatically classified as structural modes):")
    for fp, value in peaks_y:
        print(f"  {fp:.4f} Hz, plotted value={value:.6g}")
    print("Prominent X-axis frequency candidates:")
    for fp, value in peaks_x:
        print(f"  {fp:.4f} Hz, plotted value={value:.6g}")
    if f_env is not None:
        print("Envelope modulation frequency candidates (normalized Lomb–Scargle power):")
        for fp, value in env_peaks:
            print(f"  {fp:.4f} Hz, normalized power={value:.6g}")
    else:
        print("Envelope modulation spectrum not calculated: insufficient valid envelope coverage/length.")
    print("Interpretation note: a frequency peak is a candidate, not proof of a physical mode or nonlinear interaction.")
    plt.show()


if __name__ == "__main__":
    main()
