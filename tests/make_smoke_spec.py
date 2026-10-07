"""
Writes LoChaSmoke.spec: LoCha.spec with tests/exe_smoke.py as the program
and a console for its output, so the end-to-end test runs in a build made
from exactly the same recipe as LoCha.exe. Run from the project folder.
"""

spec = open("LoCha.spec", encoding="utf-8").read()
replacements = [
    ('["run_locha.py"]', '["tests/exe_smoke.py"]'),
    ('name="LoCha",', 'name="LoChaSmoke",'),  # the app exe and the output folder
    # The test script is in tests/; let it import LoCha_app from the project folder.
    ("pathex=[],", "pathex=[SPECPATH],"),
    ("console=False,  # windowed app; output goes to %LOCALAPPDATA%\\LoCha\\locha.log", "console=True,  # test output"),
]
for old, new in replacements:
    if old not in spec:
        raise SystemExit(f"LoCha.spec changed; can't find {old!r}")
    spec = spec.replace(old, new)
open("LoChaSmoke.spec", "w", encoding="utf-8").write(spec)
print("Wrote LoChaSmoke.spec")
