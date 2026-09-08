# Data — not included

NinaPro DB2 is **not redistributed in this repository**. Download it yourself from the NinaPro
project and accept their terms:

<http://ninapro.hevs.ch/>

Reference: Atzori, M. et al. *Electromyography data for non-invasive naturally-controlled robotic
hand prostheses.* Scientific Data 1, 140053 (2014).

Unzip the per-subject archives so the tree looks like this:

```
data/DB2_s1/S1_E1_A1.mat
data/DB2_s2/S2_E1_A1.mat
...
data/DB2_s15/S15_E1_A1.mat
```

Only the **E1** files are needed. S1–S10 form the training pool, S11–S15 the held-out test
subjects. Nothing under `data/` is tracked by git except this file.
