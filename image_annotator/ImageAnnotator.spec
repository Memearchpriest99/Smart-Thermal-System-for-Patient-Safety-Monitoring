# -*- mode: python ; coding: utf-8 -*-
#
# Build the standalone annotator. Because the person recommender imports the
# project's detection package, run this from the REPO ROOT so PyInstaller can
# discover thermal_algorithms at build time:
#
#     pyinstaller image_annotator/ImageAnnotator.spec
#
# If opencv / thermal_algorithms cannot be bundled, the app still runs — the
# person recommender just disables itself; fire suggestions keep working.

import os

# Repo root = parent of this spec's folder, so `import thermal_algorithms`
# resolves at build time no matter the current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# The annotator only uses the cv2/numpy-based detectors (TatenoPipeline,
# AdaptiveThresholdDetector). PyInstaller follows those static imports on its
# own; we only need to name the leaf modules defensively. Do NOT pull the whole
# package — its torch-based detectors (mv_stgcn / thermo_x3d / mobilenet_ssd)
# would drag in torch and balloon / stall the build.
hiddenimports = [
    'thermal_algorithms.core.sensor_profile',
    'thermal_algorithms.core.types',
    'thermal_algorithms.core.base',
    'thermal_algorithms.preprocessing.base',
    'thermal_algorithms.preprocessing.tateno_pipeline',
    'thermal_algorithms.human_detection.base',
    'thermal_algorithms.human_detection.adaptive_threshold',
    'cv2', 'numpy', 'openpyxl', 'PIL',
]

a = Analysis(
    [os.path.join(SPECPATH, 'annotator.py')],
    pathex=[_REPO_ROOT],    # repo root, so `import thermal_algorithms` resolves
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'optree',
              'matplotlib', 'scipy', 'pandas', 'sklearn', 'skimage'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ImageAnnotator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-compressing OpenCV DLLs is extremely slow; skip it
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
