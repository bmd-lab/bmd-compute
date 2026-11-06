"""Minimal but real VASP output files for the step-validity tests.

``write_outputs`` writes vasprun.xml (parsed by pymatgen's ``Vasprun``),
OUTCAR (with ``free  energy   TOTEN`` lines), OSZICAR and CONTCAR for a
one-atom Si cell, with the number of ionic and electronic steps chosen by
the test. ``variant`` produces the failure modes:

    "valid"            converged within NELM and NSW
    "nelm"             final ionic step used all NELM electronic steps
    "nsw"              NSW ionic steps used (relaxation not converged)
    "nan"              final energy NaN in vasprun.xml and OUTCAR
    "truncated"        vasprun.xml cut off half way (killed run)
    "missing_vasprun"  no vasprun.xml
    "missing_outcar"   no OUTCAR
    "no_toten"         OUTCAR without a TOTEN line
    "mismatch"         OUTCAR's last TOTEN differs from vasprun.xml
    "missing_contcar"  no CONTCAR
    "bad_contcar"      CONTCAR is not a structure
    "no_energy"        vasprun.xml final step without e_fr_energy
"""

from pathlib import Path

STEP_LIMITS = {1: (65, 60), 2: (55, 60), 3: (65, 60), 4: (99, 200), 5: (0, 200)}


def vasprun_xml(*, nsw, nelm, ionic_scsteps, energies, a=3.0, ibrion=2, isif=3, truncated=False):
    def structure(name):
        return f'''  <structure name="{name}">
   <crystal>
    <varray name="basis" >
     <v>  {a:.6f} 0.0 0.0 </v>
     <v>  0.0 {a:.6f} 0.0 </v>
     <v>  0.0 0.0 {a:.6f} </v>
    </varray>
    <i name="volume">  {a**3:.6f} </i>
    <varray name="rec_basis" >
     <v> {1/a:.6f} 0.0 0.0 </v>
     <v> 0.0 {1/a:.6f} 0.0 </v>
     <v> 0.0 0.0 {1/a:.6f} </v>
    </varray>
   </crystal>
   <varray name="positions" >
    <v>       0.00000000       0.00000000       0.00000000 </v>
   </varray>
  </structure>
'''
    calcs = []
    for n_sc, energy in zip(ionic_scsteps, energies):
        sc = "".join(f'''   <scstep>
    <energy>
     <i name="e_fr_energy">  {energy:.8f} </i>
     <i name="e_wo_entrp">  {energy:.8f} </i>
     <i name="e_0_energy">  {energy:.8f} </i>
    </energy>
   </scstep>
''' for _ in range(n_sc))
        calcs.append(f'''  <calculation>
{sc}{structure("")}   <varray name="forces" >
    <v>       0.00000000       0.00000000       0.00000000 </v>
   </varray>
   <varray name="stress" >
    <v>       0.0 0.0 0.0 </v>
    <v>       0.0 0.0 0.0 </v>
    <v>       0.0 0.0 0.0 </v>
   </varray>
   <energy>
    <i name="e_fr_energy">  {energy:.8f} </i>
    <i name="e_wo_entrp">  {energy:.8f} </i>
    <i name="e_0_energy">  {energy:.8f} </i>
   </energy>
  </calculation>
''')
    text = f'''<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <generator>
  <i name="program" type="string">vasp </i>
  <i name="version" type="string">6.4.2  </i>
  <i name="subversion" type="string">fake </i>
  <i name="platform" type="string">LinuxIFC </i>
  <i name="date" type="string">2026 11 06 </i>
  <i name="time" type="string">12:00:00 </i>
 </generator>
 <incar>
  <i type="int" name="NSW">     {nsw}</i>
  <i type="int" name="NELM">    {nelm}</i>
  <i type="int" name="IBRION">     {ibrion}</i>
  <i type="int" name="ISIF">     {isif}</i>
 </incar>
 <kpoints>
  <generation param="Gamma">
   <v type="int" name="divisions">       2        2        2 </v>
  </generation>
  <varray name="kpointlist" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
  </varray>
  <varray name="weights" >
   <v>       1.00000000 </v>
  </varray>
 </kpoints>
 <parameters>
  <separator name="electronic" >
   <i type="string" name="PREC">accurate</i>
   <separator name="electronic convergence" >
    <i type="int" name="NELM">    {nelm}</i>
   </separator>
  </separator>
  <separator name="ionic" >
   <i type="int" name="NSW">     {nsw}</i>
   <i type="int" name="IBRION">     {ibrion}</i>
   <i type="int" name="ISIF">     {isif}</i>
  </separator>
 </parameters>
 <atominfo>
  <atoms>       1 </atoms>
  <types>       1 </types>
  <array name="atoms" >
   <dimension dim="1">ion</dimension>
   <field type="string">element</field>
   <field type="int">atomtype</field>
   <set>
    <rc><c>Si</c><c>   1</c></rc>
   </set>
  </array>
  <array name="atomtypes" >
   <dimension dim="1">type</dimension>
   <field type="int">atomspertype</field>
   <field type="string">element</field>
   <field>mass</field>
   <field>valence</field>
   <field type="string">pseudopotential</field>
   <set>
    <rc><c>   1</c><c>Si</c><c>     28.08500000</c><c>      4.00000000</c><c>  PAW_PBE Si 05Jan2001                   </c></rc>
   </set>
  </array>
 </atominfo>
{structure("initialpos")}{"".join(calcs)}{structure("finalpos")} </modeling>
'''
    if truncated:
        text = text[: len(text) // 2]
    return text


def _remove_e_fr(text: str) -> str:
    # Drop the final ionic step's e_fr_energy (the last <energy> block of the last calculation).
    head, sep, tail = text.rpartition('<i name="e_fr_energy">')
    end = tail.index("</i>") + len("</i>")
    return head + tail[end:]


def write_outputs(folder: Path, step: int, energy: float, variant: str = "valid", label: str = "") -> None:
    folder = Path(folder)
    nsw, nelm = STEP_LIMITS.get(step, (10, 60))
    if variant == "nsw":
        ionic = max(nsw, 1)
        scsteps = [8] * ionic
    elif nsw == 0:
        ionic, scsteps = 1, [12]
    else:
        ionic, scsteps = 3, [12, 9, 7]
    if variant == "nelm":
        scsteps[-1] = nelm
    energies = [energy + 0.1 * (ionic - 1 - i) for i in range(ionic)]
    if variant == "nan":
        energies[-1] = float("nan")
    text = vasprun_xml(nsw=nsw, nelm=nelm, ionic_scsteps=scsteps, energies=energies)
    if variant == "truncated":
        text = text[: len(text) // 2]
    if variant == "no_energy":
        text = _remove_e_fr(text)
    if variant != "missing_vasprun":
        (folder / "vasprun.xml").write_text(text)
    outcar = [" vasp.6.4.2 fake OUTCAR"]
    if variant != "no_toten":
        for value in energies:
            shown = value + (1.0 if variant == "mismatch" and value is energies[-1] else 0.0)
            text_value = "NaN" if shown != shown else f"{shown:.8f}"
            outcar.append(f"  free  energy   TOTEN  =       {text_value} eV")
    outcar.append(" General timing and accounting informations for this job:")
    if variant != "missing_outcar":
        (folder / "OUTCAR").write_text("\n".join(outcar) + "\n")
    final = energies[-1]
    (folder / "OSZICAR").write_text(
        "       N       E                     dE             d eps       ncg     rms          rms(c)\n"
        f"   1 F= {final:.8E} E0= {final:.8E}  d E =-.1E-03\n"
    )
    poscar = (folder / "POSCAR").read_text().splitlines()
    poscar[0] = f"{poscar[0]} {label}".rstrip()
    if variant == "bad_contcar":
        (folder / "CONTCAR").write_text("not a structure\n")
    elif variant != "missing_contcar":
        (folder / "CONTCAR").write_text("\n".join(poscar) + "\n")
