"""
ATHENA: Automatically Tracking Hands Expertly with No Annotations.

A multi-camera 3D hand and body tracking pipeline built on MediaPipe and
optionally HaMeR.

Modules
-------
labels2d
    Phase 1 -- per-camera 2D landmark detection and cross-camera hand
    reassignment.
triangulaterefine
    Phase 2 -- multi-camera DLT triangulation, temporal smoothing, hand-swap
    correction, and visualisation.
visualization
    Shared skeleton topology, colours, and mesh-rendering helpers used by
    both phases.
hamer_hands
    HaMeR hand-mesh regression backend (optional, requires ``torch``).
montage
    Multi-camera video montage creation.
athena
    Tkinter GUI launcher.
"""

__version__ = "1.0"
