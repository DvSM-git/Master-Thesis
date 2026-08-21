# Angrist & Evans (1998) — data provenance and verification

Data for the replication of:

> Angrist, Joshua D. and William N. Evans (1998). "Children and Their Parents'
> Labor Supply: Evidence from Exogenous Variation in Family Size."
> *American Economic Review* 88(3), 450–477.

## Files

| File | Contents |
|---|---|
| `Angrist-ChildrenParentsLabor-1998.pdf` | The published article (JSTOR copy, downloaded 2026-07-24). |
| `m_d_806.sas7bdat` | Raw 1980 Census 5% PUMS extract: 927,267 women aged 20–60 with one or more own children in the household, with the children's and (where present) the matched husband's variables. 85 columns; most values stored as byte strings. |
| `m_d_903.sas7bdat` | Raw 1990 Census 5% PUMS extract, same design: 974,693 women aged 20–60 with one or more children. 69 columns; includes PUMS person weights (`PWGTM1`), which the paper uses for all 1990 calculations. |

## Provenance

Both SAS files are the authors' original posted extracts ("mom and dad" files),
distributed via Josh Angrist's data archive at MIT
(originally `http://economics.mit.edu/faculty/angrist/data1/data/angev98`,
now the "Angrist Data Archive" on his MIT faculty page). Downloaded 2026-07-24.
The same files are redistributed by the *Journal of Applied Econometrics* data
archive for Farbmacher, Guber & Vikström (2018), which documents this
provenance in its readme
(`http://qed.econ.queensu.ca/jae/datasets/farbmacher002/`, plain http).

## Coding conventions (1980 file)

- Values are byte strings (e.g. `b'01'`) or floats; decode before use.
- Child sex codes: `0` = boy, `1` = girl (`SEXK`, `SEX2ND`).
- Children's ages are in quarters at census day (`AGEQK`, `AGEQ2ND`, `AGEQ3RD`).
- `A*` variables are Census allocation (imputation) flags: `AAGE`, `AQTRBRTH`,
  `ASEX`, `AAGE2ND`, `AQTR2ND`, `ASEX2ND` (`1` = value was allocated).
  `ASEX`/`ASEX2ND` are all zero in the posted extract.
- Mother ("M") suffix: `AGEM`, `WEEKSM`, `HOURSM`, `INCOME1M` (wage/salary),
  `INCOME2M` (self-employment, can be negative), `FAMINC`.
  Husband/"dad" ("D") suffix variables are missing when no husband is present.
- Incomes are 1979 dollars, top-coded at $75,000; the paper converts to 1995
  dollars with the CPI factor 2.099173554.

## Analysis sample construction (1980, all women)

The published article's data appendix is in the NBER working paper (Angrist &
Evans 1996, WP 5778), which is not in this folder. The selection rules used
here follow the replication do-file `AngristEvans1980_jae.do` from the
Farbmacher–Guber–Vikström JAE archive, which works from this exact file:

```
21 <= AGEM <= 35
KIDCOUNT >= 2
AGEQ2ND > 4                     (second child more than a year old)
age at first birth >= 15, where
    ageqm   = 4*(80 - YOBM) - QTRBTHM - 1
    agefstm = floor((ageqm - AGEQK) / 4)
AAGE = AQTRBRTH = AAGE2ND = ASEX = ASEX2ND = 0   (no allocated values)
```

This yields **n = 394,840** against the published 394,835 — a known,
unresolved five-observation discrepancy of the posted archive (Farbmacher
et al. also report 394,840 in print).

Implementation: `Code/replication_ae98.py` (`build_sample`), with cached
intermediates in `Code/output/ae98_raw_1980.pkl` and
`Code/output/ae98_sample_1980.pkl`.

## Verification against the published tables

Checks run on 2026-07-24 (see `Code/replication_ae98.py` console output and
`Code/output/ae98_replication.csv`):

- **1980 file** — reproduces at published precision: Table 2 means/SDs
  (e.g. labor income 7,161 vs 7,160, SD 10,804 exact; worked-for-pay 0.565;
  children ever born 2.55/0.81), Table 3 fractions (0.432 / 0.372 by sex mix),
  Table 5 first stage (0.0595 vs 0.0600, same at reported rounding) and all
  five Wald rows (e.g. Worked for pay −0.1318 (0.0263) vs −0.133 (0.026)).
  Conclusion: this is the correct data for the paper.
- **1990 file** — structural checks only (row count, age range, weights
  present on all rows, 7,611 multiple second births ≈ the 1.2% the paper
  reports). Not used in the thesis replication because the paper's 1990
  results require PUMS person weights (outside the thesis's i.i.d.
  framework) and twins are mismeasured without quarter of birth.
