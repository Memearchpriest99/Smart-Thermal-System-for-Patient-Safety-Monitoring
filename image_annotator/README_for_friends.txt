Thermal Image Annotator
=======================

WHAT IT IS
  A tool for drawing YOLO bounding boxes on the 3-channel thermal recordings
  (fire / person) and marking per-frame contact ("touch"). It renders the
  images directly from the thermal .npz data.

HOW TO RUN
  Just double-click  ImageAnnotator.exe
  (Windows 10/11, 64-bit. No Python or install needed. Windows SmartScreen may
   warn about an unsigned app the first time -> "More info" -> "Run anyway".)

HOW TO USE
  1. Click "Open Dataset Directory" and pick a SESSION folder, e.g.
        .../waveshare_organized/2ppl_hug
     (it must contain ch0_frames/, ch1_frames/, ch2_frames/ and the
      ch0_raw_data.npz / ch1_raw_data.npz / ch2_raw_data.npz files).
  2. Pick a class (fire / person) and drag a box on any of the 3 views.
     Arrow keys (or Prev/Next) move through frames; boxes auto-save.
  3. TOUCH button (or press T): marks contact for the current frame across all
     3 views. Green = 1, grey = 0.
  4. PROCESSING: choose how the image is shown -
        None (raw) | Gaussian (with sigma) | Tateno (background-subtracted).
  5. Resize W x H + Apply: render resolution (bicubic). Default 320x240.
  6. Suggest (press S): shows automatic ghost boxes (fire + person).
     Accept (press A) commits the active view's suggestions.
  7. Export Labels: writes labels.xlsx (per-frame, per-channel) and
     contact_labels.csv into the session folder. (Scroll the left panel down if
     you don't see the button.)

NOTES
  - "Tateno" mode and the person suggestions need an "empty_room" (or
    "calibrate_room") recording sitting NEXT TO the session you open, inside the
    same dataset folder. Without it, Tateno shows the raw image and only fire is
    suggested.
  - Saved labels are normalized YOLO, so the Resize resolution does not change
    them.
