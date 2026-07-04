"""Live demo application — Smart Thermal System for Patient Safety Monitoring.

Real-time 3-camera thermal UI with person / fire bounding-box overlays and
contact (touch) detection. Torch-free at runtime: MobileNet-SSD and
Thermo-X3D run via onnxruntime, FireSVM via scikit-learn.

Run:
    python -m demo.app --replay datasets/waveshare_work/2ppl_fight   # dev
    python -m demo.app --live --config demo/cams_pi.json             # Pi
"""
