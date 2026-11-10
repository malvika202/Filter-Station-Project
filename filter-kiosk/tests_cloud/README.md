# The cloud tests (for reference)

These are the tests Claude ran before the Pi existed (results in ../TEST_REPORT.md). They need the
Goldilocks project kit (its hallway scene and public test photos) set up at the paths inside them,
so they won't run on the Pi as they are. They're here so the team can see exactly what was tested.

- goldilocks_gate_test.py: team rules inside Goldilocks v3 (hair only on faces that pass, blur, head turn)
- test_adapters.py: all four filters inside the kiosk; every far face blurred
- test_switching.py: frame-by-frame switching, fast taps, crash fallback, privacy during switches
- test_ui.py, test_doubletap.py: the page in Chrome with touch taps and mouse clicks (AC1 to AC8)
- test_calibration.py: calibrating through the staff panel under each filter
- visor_sweep.py: The Visor v1 vs v2 at different distances
