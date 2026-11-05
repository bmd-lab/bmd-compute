# Real USPEX 25.0 output: Si8 pilot

`results1/` is real output of the Phase 0 Si8 pilot run on POWER on
4 November 2026 (issue #59): fixed composition Si8 at 0 GPa, population 8,
4 generations, 3 VASP relaxation steps with pilot (not BMD-policy) INCARs.
It is a toy system and contains no research data and no POTCAR content.

The files are a byte-for-byte copy of
`reference/uspex_pilot/si_bridge/sample_results1/` at commit `3cdeebc`
(branch `claude/confident-ride-qqpxp5`, PR #76). They are copied here so the
parser tests do not depend on that PR. `results1/README.md` is the pilot's own
description. Its relative links (`../`, `../Specific/`) point into
`reference/uspex_pilot/si_bridge/` and do not resolve from here.

Tests in `tests/test_structure_search_results.py` use these files directly.
They also copy them into a temporary directory and modify the copies (truncate
a line, add a `results2/`, remove a generation), so these files stay unchanged.
