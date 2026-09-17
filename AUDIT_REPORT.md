# AUDIT_REPORT.md — Forensic Audit of All Existing Experiments

**Total runs audited:** 199

**Audit date:** September 17, 2026


---

## Critical Finding: Stage 2 is NOT Modular Addition

**The current manuscript claims Stage 2 tests the additive group (modular addition).**
This is **false**. Config comparison confirms:

- Stage 1 and Stage 2 use the **same `dataset_sha256`** (`ebae03a7...`)
- Same `p=54721`, same `low_bit=0`, same `runner=paper_transformer`
- Only differences: optimizer (adam→adamw), WD (0→0.3), epochs (2000→100000)
- The `generate_takhanov_paper_data()` function computes `(a*x) % p` — this is DLP parity, not modular addition

**Stage 2 grokked because of longer training + weight decay, NOT because of a different group structure.**
The entire 'group-structure-dependent grokking' narrative in the current paper is invalid.


---

## Summary by Stage

| Stage | Description | Runs | Phases |
|-------|-------------|-----|--------|
| stage00 | Stage 0: MLP Replication | 90 | direct_generalization=14, grokking=2, memorization_only=62, partial=12 |
| stage00b | Stage 0b: Theorem Uniform | 9 | direct_generalization=3, memorization_only=4, partial=2 |
| stage01 | Stage 1: Architecture Bridge | 18 | underfit=18 |
| stage02 | Stage 2: Grokking Bridge (NOT modular addition) | 18 | direct_generalization=2, grokking=3, memorization_only=11, underfit=2 |
| stage03 | Stage 3: 2×2 Discovery | 64 | memorization_only=60, partial=4 |


## Stage 0: MLP Replication

**Runs:** 90

**Dataset hashes:** 90 distinct hashes

| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |
|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|
| R0_b16_rep0_bit0 | Unknown runner: paper_mlp | 53077 | 16 | 101 | adam | 0 | 2000 | direct_generalization | 500 | 300 | -200 | 1.0000 | 1.0000 |
| R0_b16_rep0_bit1 | Unknown runner: paper_mlp | 53077 | 16 | 101 | adam | 0 | 2000 | direct_generalization | 600 | 500 | -100 | 1.0000 | 0.9893 |
| R0_b16_rep0_bit2 | Unknown runner: paper_mlp | 53077 | 16 | 101 | adam | 0 | 2000 | grokking | 600 | 700 | 100 | 1.0000 | 0.9033 |
| R0_b16_rep1_bit0 | Unknown runner: paper_mlp | 42061 | 16 | 202 | adam | 0 | 2000 | direct_generalization | 200 | 200 | 0 | 1.0000 | 1.0000 |
| R0_b16_rep1_bit1 | Unknown runner: paper_mlp | 42061 | 16 | 202 | adam | 0 | 2000 | direct_generalization | 300 | 300 | 0 | 1.0000 | 1.0000 |
| R0_b16_rep1_bit2 | Unknown runner: paper_mlp | 42061 | 16 | 202 | adam | 0 | 2000 | partial | 700 | — | — | 1.0000 | 0.5953 |
| R0_b16_rep2_bit0 | Unknown runner: paper_mlp | 61403 | 16 | 303 | adam | 0 | 2000 | direct_generalization | 700 | 400 | -300 | 1.0000 | 1.0000 |
| R0_b16_rep2_bit1 | Unknown runner: paper_mlp | 61403 | 16 | 303 | adam | 0 | 2000 | direct_generalization | 300 | 300 | 0 | 1.0000 | 1.0000 |
| R0_b16_rep2_bit2 | Unknown runner: paper_mlp | 61403 | 16 | 303 | adam | 0 | 2000 | grokking | 500 | 1400 | 900 | 1.0000 | 0.9147 |
| R0_b16_rep3_bit0 | Unknown runner: paper_mlp | 62233 | 16 | 404 | adam | 0 | 2000 | direct_generalization | 300 | 300 | 0 | 1.0000 | 0.9967 |
| R0_b16_rep3_bit1 | Unknown runner: paper_mlp | 62233 | 16 | 404 | adam | 0 | 2000 | direct_generalization | 200 | 200 | 0 | 1.0000 | 1.0000 |
| R0_b16_rep3_bit2 | Unknown runner: paper_mlp | 62233 | 16 | 404 | adam | 0 | 2000 | direct_generalization | 300 | 300 | 0 | 1.0000 | 0.9947 |
| R0_b16_rep4_bit0 | Unknown runner: paper_mlp | 64171 | 16 | 505 | adam | 0 | 2000 | direct_generalization | 200 | 200 | 0 | 1.0000 | 1.0000 |
| R0_b16_rep4_bit1 | Unknown runner: paper_mlp | 64171 | 16 | 505 | adam | 0 | 2000 | direct_generalization | 500 | 400 | -100 | 1.0000 | 0.9640 |
| R0_b16_rep4_bit2 | Unknown runner: paper_mlp | 64171 | 16 | 505 | adam | 0 | 2000 | partial | 500 | — | — | 1.0000 | 0.8107 |
| R0_b18_rep0_bit0 | Unknown runner: paper_mlp | 212353 | 18 | 101 | adam | 0 | 2000 | partial | 600 | — | — | 1.0000 | 0.7453 |
| R0_b18_rep0_bit1 | Unknown runner: paper_mlp | 212353 | 18 | 101 | adam | 0 | 2000 | partial | 500 | — | — | 1.0000 | 0.5693 |
| R0_b18_rep0_bit2 | Unknown runner: paper_mlp | 212353 | 18 | 101 | adam | 0 | 2000 | partial | 600 | — | — | 1.0000 | 0.6773 |
| R0_b18_rep1_bit0 | Unknown runner: paper_mlp | 168247 | 18 | 202 | adam | 0 | 2000 | partial | 700 | — | — | 1.0000 | 0.5660 |
| R0_b18_rep1_bit1 | Unknown runner: paper_mlp | 168247 | 18 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5207 |
| R0_b18_rep1_bit2 | Unknown runner: paper_mlp | 168247 | 18 | 202 | adam | 0 | 2000 | direct_generalization | 600 | 400 | -200 | 1.0000 | 0.9793 |
| R0_b18_rep2_bit0 | Unknown runner: paper_mlp | 245587 | 18 | 303 | adam | 0 | 2000 | partial | 600 | — | — | 1.0000 | 0.7820 |
| R0_b18_rep2_bit1 | Unknown runner: paper_mlp | 245587 | 18 | 303 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5093 |
| R0_b18_rep2_bit2 | Unknown runner: paper_mlp | 245587 | 18 | 303 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.5353 |
| R0_b18_rep3_bit0 | Unknown runner: paper_mlp | 248903 | 18 | 404 | adam | 0 | 2000 | partial | 500 | — | — | 1.0000 | 0.7500 |
| R0_b18_rep3_bit1 | Unknown runner: paper_mlp | 248903 | 18 | 404 | adam | 0 | 2000 | partial | 600 | — | — | 1.0000 | 0.7887 |
| R0_b18_rep3_bit2 | Unknown runner: paper_mlp | 248903 | 18 | 404 | adam | 0 | 2000 | partial | 700 | — | — | 1.0000 | 0.6340 |
| R0_b18_rep4_bit0 | Unknown runner: paper_mlp | 256687 | 18 | 505 | adam | 0 | 2000 | direct_generalization | 600 | 600 | 0 | 1.0000 | 0.9873 |
| R0_b18_rep4_bit1 | Unknown runner: paper_mlp | 256687 | 18 | 505 | adam | 0 | 2000 | partial | 600 | — | — | 1.0000 | 0.7433 |
| R0_b18_rep4_bit2 | Unknown runner: paper_mlp | 256687 | 18 | 505 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.5413 |
| R0_b20_rep0_bit0 | Unknown runner: paper_mlp | 849203 | 20 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5033 |
| R0_b20_rep0_bit1 | Unknown runner: paper_mlp | 849203 | 20 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5167 |
| R0_b20_rep0_bit2 | Unknown runner: paper_mlp | 849203 | 20 | 101 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5100 |
| R0_b20_rep1_bit0 | Unknown runner: paper_mlp | 672977 | 20 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4920 |
| R0_b20_rep1_bit1 | Unknown runner: paper_mlp | 672977 | 20 | 202 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.4753 |
| R0_b20_rep1_bit2 | Unknown runner: paper_mlp | 672977 | 20 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4980 |
| R0_b20_rep2_bit0 | Unknown runner: paper_mlp | 982321 | 20 | 303 | adam | 0 | 2000 | direct_generalization | 500 | 400 | -100 | 1.0000 | 0.9953 |
| R0_b20_rep2_bit1 | Unknown runner: paper_mlp | 982321 | 20 | 303 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.5280 |
| R0_b20_rep2_bit2 | Unknown runner: paper_mlp | 982321 | 20 | 303 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5027 |
| R0_b20_rep3_bit0 | Unknown runner: paper_mlp | 995591 | 20 | 404 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.4920 |
| R0_b20_rep3_bit1 | Unknown runner: paper_mlp | 995591 | 20 | 404 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4913 |
| R0_b20_rep3_bit2 | Unknown runner: paper_mlp | 995591 | 20 | 404 | adam | 0 | 2000 | memorization_only | 700 | — | — | 1.0000 | 0.5267 |
| R0_b20_rep4_bit0 | Unknown runner: paper_mlp | 1026757 | 20 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5080 |
| R0_b20_rep4_bit1 | Unknown runner: paper_mlp | 1026757 | 20 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5033 |
| R0_b20_rep4_bit2 | Unknown runner: paper_mlp | 1026757 | 20 | 505 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5040 |
| R0_b22_rep0_bit0 | Unknown runner: paper_mlp | 3396829 | 22 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4913 |
| R0_b22_rep0_bit1 | Unknown runner: paper_mlp | 3396829 | 22 | 101 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5247 |
| R0_b22_rep0_bit2 | Unknown runner: paper_mlp | 3396829 | 22 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4987 |
| R0_b22_rep1_bit0 | Unknown runner: paper_mlp | 2691893 | 22 | 202 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5000 |
| R0_b22_rep1_bit1 | Unknown runner: paper_mlp | 2691893 | 22 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4740 |
| R0_b22_rep1_bit2 | Unknown runner: paper_mlp | 2691893 | 22 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4913 |
| R0_b22_rep2_bit0 | Unknown runner: paper_mlp | 3929267 | 22 | 303 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5060 |
| R0_b22_rep2_bit1 | Unknown runner: paper_mlp | 3929267 | 22 | 303 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4947 |
| R0_b22_rep2_bit2 | Unknown runner: paper_mlp | 3929267 | 22 | 303 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5193 |
| R0_b22_rep3_bit0 | Unknown runner: paper_mlp | 3982373 | 22 | 404 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5200 |
| R0_b22_rep3_bit1 | Unknown runner: paper_mlp | 3982373 | 22 | 404 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5013 |
| R0_b22_rep3_bit2 | Unknown runner: paper_mlp | 3982373 | 22 | 404 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5013 |
| R0_b22_rep4_bit0 | Unknown runner: paper_mlp | 4106987 | 22 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5033 |
| R0_b22_rep4_bit1 | Unknown runner: paper_mlp | 4106987 | 22 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4960 |
| R0_b22_rep4_bit2 | Unknown runner: paper_mlp | 4106987 | 22 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4847 |
| R0_b24_rep0_bit0 | Unknown runner: paper_mlp | 13587221 | 24 | 101 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5140 |
| R0_b24_rep0_bit1 | Unknown runner: paper_mlp | 13587221 | 24 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4853 |
| R0_b24_rep0_bit2 | Unknown runner: paper_mlp | 13587221 | 24 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5093 |
| R0_b24_rep1_bit0 | Unknown runner: paper_mlp | 10767551 | 24 | 202 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4920 |
| R0_b24_rep1_bit1 | Unknown runner: paper_mlp | 10767551 | 24 | 202 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.4973 |
| R0_b24_rep1_bit2 | Unknown runner: paper_mlp | 10767551 | 24 | 202 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5067 |
| R0_b24_rep2_bit0 | Unknown runner: paper_mlp | 15717071 | 24 | 303 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5000 |
| R0_b24_rep2_bit1 | Unknown runner: paper_mlp | 15717071 | 24 | 303 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4787 |
| R0_b24_rep2_bit2 | Unknown runner: paper_mlp | 15717071 | 24 | 303 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5127 |
| R0_b24_rep3_bit0 | Unknown runner: paper_mlp | 15929453 | 24 | 404 | adam | 0 | 2000 | partial | 400 | — | — | 1.0000 | 0.7267 |
| R0_b24_rep3_bit1 | Unknown runner: paper_mlp | 15929453 | 24 | 404 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5240 |
| R0_b24_rep3_bit2 | Unknown runner: paper_mlp | 15929453 | 24 | 404 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4967 |
| R0_b24_rep4_bit0 | Unknown runner: paper_mlp | 16427981 | 24 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4967 |
| R0_b24_rep4_bit1 | Unknown runner: paper_mlp | 16427981 | 24 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5073 |
| R0_b24_rep4_bit2 | Unknown runner: paper_mlp | 16427981 | 24 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4873 |
| R0_b26_rep0_bit0 | Unknown runner: paper_mlp | 54348869 | 26 | 101 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5267 |
| R0_b26_rep0_bit1 | Unknown runner: paper_mlp | 54348869 | 26 | 101 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5000 |
| R0_b26_rep0_bit2 | Unknown runner: paper_mlp | 54348869 | 26 | 101 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5180 |
| R0_b26_rep1_bit0 | Unknown runner: paper_mlp | 43070173 | 26 | 202 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5020 |
| R0_b26_rep1_bit1 | Unknown runner: paper_mlp | 43070173 | 26 | 202 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5067 |
| R0_b26_rep1_bit2 | Unknown runner: paper_mlp | 43070173 | 26 | 202 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4913 |
| R0_b26_rep2_bit0 | Unknown runner: paper_mlp | 62868209 | 26 | 303 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5233 |
| R0_b26_rep2_bit1 | Unknown runner: paper_mlp | 62868209 | 26 | 303 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.4940 |
| R0_b26_rep2_bit2 | Unknown runner: paper_mlp | 62868209 | 26 | 303 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4940 |
| R0_b26_rep3_bit0 | Unknown runner: paper_mlp | 63717821 | 26 | 404 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.4980 |
| R0_b26_rep3_bit1 | Unknown runner: paper_mlp | 63717821 | 26 | 404 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5180 |
| R0_b26_rep3_bit2 | Unknown runner: paper_mlp | 63717821 | 26 | 404 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.5067 |
| R0_b26_rep4_bit0 | Unknown runner: paper_mlp | 65711689 | 26 | 505 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.5080 |
| R0_b26_rep4_bit1 | Unknown runner: paper_mlp | 65711689 | 26 | 505 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4880 |
| R0_b26_rep4_bit2 | Unknown runner: paper_mlp | 65711689 | 26 | 505 | adam | 0 | 2000 | memorization_only | 400 | — | — | 1.0000 | 0.4867 |


## Stage 0b: Theorem Uniform

**Runs:** 9

**Dataset hashes:** 9 distinct hashes

| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |
|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|
| R0b_b16_s123 | Unknown runner: paper_mlp | 60631 | 16 | 123 | adam | 0 | 2000 | direct_generalization | 300 | 300 | 0 | 1.0000 | 0.9700 |
| R0b_b16_s42 | Unknown runner: paper_mlp | 54721 | 16 | 42 | adam | 0 | 2000 | direct_generalization | 500 | 500 | 0 | 1.0000 | 0.9627 |
| R0b_b16_s456 | Unknown runner: paper_mlp | 49429 | 16 | 456 | adam | 0 | 2000 | direct_generalization | 400 | 300 | -100 | 1.0000 | 0.9820 |
| R0b_b20_s123 | Unknown runner: paper_mlp | 970061 | 20 | 123 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.4833 |
| R0b_b20_s42 | Unknown runner: paper_mlp | 875503 | 20 | 42 | adam | 0 | 2000 | partial | 500 | — | — | 1.0000 | 0.7440 |
| R0b_b20_s456 | Unknown runner: paper_mlp | 790871 | 20 | 456 | adam | 0 | 2000 | partial | 500 | — | — | 1.0000 | 0.5780 |
| R0b_b24_s123 | Unknown runner: paper_mlp | 15520909 | 24 | 123 | adam | 0 | 2000 | memorization_only | 600 | — | — | 1.0000 | 0.5113 |
| R0b_b24_s42 | Unknown runner: paper_mlp | 14007971 | 24 | 42 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4947 |
| R0b_b24_s456 | Unknown runner: paper_mlp | 12653903 | 24 | 456 | adam | 0 | 2000 | memorization_only | 500 | — | — | 1.0000 | 0.4967 |


## Stage 1: Architecture Bridge

**Runs:** 18

**Dataset hashes:** 18 distinct hashes

| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |
|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|
| R1_b16_s123 | DLP parity bit 0 (p=60631, b=16bit, inpu | 60631 | 16 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.8329 | 0.5247 |
| R1_b16_s42 | DLP parity bit 0 (p=54721, b=16bit, inpu | 54721 | 16 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.5840 | 0.5207 |
| R1_b16_s456 | DLP parity bit 0 (p=49429, b=16bit, inpu | 49429 | 16 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.5743 | 0.5300 |
| R1_b18_s123 | DLP parity bit 0 (p=242519, b=18bit, inp | 242519 | 18 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.6714 | 0.5087 |
| R1_b18_s42 | DLP parity bit 0 (p=218887, b=18bit, inp | 218887 | 18 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.6689 | 0.5120 |
| R1_b18_s456 | DLP parity bit 0 (p=197741, b=18bit, inp | 197741 | 18 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.5794 | 0.4873 |
| R1_b20_s123 | DLP parity bit 0 (p=970061, b=20bit, inp | 970061 | 20 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.5997 | 0.4847 |
| R1_b20_s42 | DLP parity bit 0 (p=875503, b=20bit, inp | 875503 | 20 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.6131 | 0.4900 |
| R1_b20_s456 | DLP parity bit 0 (p=790871, b=20bit, inp | 790871 | 20 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.8886 | 0.4893 |
| R1_b22_s123 | DLP parity bit 0 (p=3880241, b=22bit, in | 3880241 | 22 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.7340 | 0.5053 |
| R1_b22_s42 | DLP parity bit 0 (p=3501989, b=22bit, in | 3501989 | 22 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.5271 | 0.5193 |
| R1_b22_s456 | DLP parity bit 0 (p=3163471, b=22bit, in | 3163471 | 22 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.5606 | 0.5067 |
| R1_b24_s123 | DLP parity bit 0 (p=15520909, b=24bit, i | 15520909 | 24 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.5326 | 0.4813 |
| R1_b24_s42 | DLP parity bit 0 (p=14007971, b=24bit, i | 14007971 | 24 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.7934 | 0.4840 |
| R1_b24_s456 | DLP parity bit 0 (p=12653903, b=24bit, i | 12653903 | 24 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.5311 | 0.4967 |
| R1_b26_s123 | DLP parity bit 0 (p=62083639, b=26bit, i | 62083639 | 26 | 123 | adam | 0 | 2000 | underfit | — | — | — | 0.5611 | 0.4960 |
| R1_b26_s42 | DLP parity bit 0 (p=56031791, b=26bit, i | 56031791 | 26 | 42 | adam | 0 | 2000 | underfit | — | — | — | 0.7329 | 0.5007 |
| R1_b26_s456 | DLP parity bit 0 (p=50615531, b=26bit, i | 50615531 | 26 | 456 | adam | 0 | 2000 | underfit | — | — | — | 0.5191 | 0.5040 |


## Stage 2: Grokking Bridge (NOT modular addition)

**Runs:** 18

**Dataset hashes:** 18 distinct hashes

| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |
|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|
| R2_b16_s123 | DLP parity bit 0 (p=60631, b=16bit, inpu | 60631 | 16 | 123 | adamw | 0.3 | 100000 | grokking | 2550 | 10400 | 7850 | 0.9840 | 0.9820 |
| R2_b16_s42 | DLP parity bit 0 (p=54721, b=16bit, inpu | 54721 | 16 | 42 | adamw | 0.3 | 100000 | direct_generalization | 1750 | 1550 | -200 | 0.9937 | 0.9880 |
| R2_b16_s456 | DLP parity bit 0 (p=49429, b=16bit, inpu | 49429 | 16 | 456 | adamw | 0.3 | 100000 | direct_generalization | 1500 | 1150 | -350 | 0.9603 | 0.9627 |
| R2_b18_s123 | DLP parity bit 0 (p=242519, b=18bit, inp | 242519 | 18 | 123 | adamw | 0.3 | 100000 | memorization_only | 2100 | — | — | 0.9980 | 0.5047 |
| R2_b18_s42 | DLP parity bit 0 (p=218887, b=18bit, inp | 218887 | 18 | 42 | adamw | 0.3 | 100000 | memorization_only | 4950 | — | — | 0.9980 | 0.5140 |
| R2_b18_s456 | DLP parity bit 0 (p=197741, b=18bit, inp | 197741 | 18 | 456 | adamw | 0.3 | 100000 | grokking | 2300 | 24250 | 21950 | 0.9969 | 0.4907 |
| R2_b20_s123 | DLP parity bit 0 (p=970061, b=20bit, inp | 970061 | 20 | 123 | adamw | 0.3 | 100000 | grokking | 3700 | 10000 | 6300 | 0.9986 | 0.5013 |
| R2_b20_s42 | DLP parity bit 0 (p=875503, b=20bit, inp | 875503 | 20 | 42 | adamw | 0.3 | 100000 | memorization_only | 4050 | — | — | 0.9986 | 0.4727 |
| R2_b20_s456 | DLP parity bit 0 (p=790871, b=20bit, inp | 790871 | 20 | 456 | adamw | 0.3 | 100000 | memorization_only | 2200 | — | — | 0.8963 | 0.4947 |
| R2_b22_s123 | DLP parity bit 0 (p=3880241, b=22bit, in | 3880241 | 22 | 123 | adamw | 0.3 | 100000 | memorization_only | 3250 | — | — | 0.9971 | 0.5113 |
| R2_b22_s42 | DLP parity bit 0 (p=3501989, b=22bit, in | 3501989 | 22 | 42 | adamw | 0.3 | 100000 | memorization_only | 3500 | — | — | 0.9997 | 0.5067 |
| R2_b22_s456 | DLP parity bit 0 (p=3163471, b=22bit, in | 3163471 | 22 | 456 | adamw | 0.3 | 100000 | memorization_only | 2050 | — | — | 0.9983 | 0.4973 |
| R2_b24_s123 | DLP parity bit 0 (p=15520909, b=24bit, i | 15520909 | 24 | 123 | adamw | 0.3 | 100000 | memorization_only | 2300 | — | — | 0.9963 | 0.4773 |
| R2_b24_s42 | DLP parity bit 0 (p=14007971, b=24bit, i | 14007971 | 24 | 42 | adamw | 0.3 | 100000 | memorization_only | 2850 | — | — | 0.9997 | 0.5053 |
| R2_b24_s456 | DLP parity bit 0 (p=12653903, b=24bit, i | 12653903 | 24 | 456 | adamw | 0.3 | 100000 | memorization_only | 2350 | — | — | 1.0000 | 0.4967 |
| R2_b26_s123 | DLP parity bit 0 (p=62083639, b=26bit, i | 62083639 | 26 | 123 | adamw | 0.3 | 100000 | underfit | — | — | — | 0.6117 | 0.4813 |
| R2_b26_s42 | DLP parity bit 0 (p=56031791, b=26bit, i | 56031791 | 26 | 42 | adamw | 0.3 | 100000 | memorization_only | 7550 | — | — | 0.9983 | 0.4893 |
| R2_b26_s456 | DLP parity bit 0 (p=50615531, b=26bit, i | 50615531 | 26 | 456 | adamw | 0.3 | 100000 | underfit | — | — | — | 0.6937 | 0.5173 |


## Stage 3: 2×2 Discovery

**Runs:** 64

**Dataset hashes:** 32 distinct hashes

| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |
|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|
| M1_b10_hidden_full_s123 | hidden_full | 1187 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 680 | — | — | 1.0000 | 0.0000 |
| M1_b10_hidden_full_s42 | hidden_full | 1187 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 660 | — | — | 1.0000 | 0.0028 |
| M1_b10_hidden_parity_s123 | hidden_parity | 1187 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1620 | — | — | 1.0000 | 0.5465 |
| M1_b10_hidden_parity_s42 | hidden_parity | 1187 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2560 | — | — | 1.0000 | 0.5324 |
| M1_b10_visible_full_s123 | visible_full | 1187 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 620 | — | — | 1.0000 | 0.0000 |
| M1_b10_visible_full_s42 | visible_full | 1187 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 700 | — | — | 1.0000 | 0.0028 |
| M1_b10_visible_parity_s123 | visible_parity | 1187 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1580 | — | — | 0.8876 | 0.4930 |
| M1_b10_visible_parity_s42 | visible_parity | 1187 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 3300 | — | — | 1.0000 | 0.4986 |
| M1_b11_hidden_full_s123 | hidden_full | 2063 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 640 | — | — | 1.0000 | 0.0000 |
| M1_b11_hidden_full_s42 | hidden_full | 2063 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 680 | — | — | 1.0000 | 0.0000 |
| M1_b11_hidden_parity_s123 | hidden_parity | 2063 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1680 | — | — | 1.0000 | 0.4919 |
| M1_b11_hidden_parity_s42 | hidden_parity | 2063 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2540 | — | — | 0.8058 | 0.5307 |
| M1_b11_visible_full_s123 | visible_full | 2063 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 720 | — | — | 1.0000 | 0.0000 |
| M1_b11_visible_full_s42 | visible_full | 2063 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 800 | — | — | 1.0000 | 0.0065 |
| M1_b11_visible_parity_s123 | visible_parity | 2063 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1380 | — | — | 1.0000 | 0.4887 |
| M1_b11_visible_parity_s42 | visible_parity | 2063 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 3140 | — | — | 0.9871 | 0.4903 |
| M1_b12_hidden_full_s123 | hidden_full | 4127 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 900 | — | — | 1.0000 | 0.0016 |
| M1_b12_hidden_full_s42 | hidden_full | 4127 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1040 | — | — | 1.0000 | 0.0000 |
| M1_b12_hidden_parity_s123 | hidden_parity | 4127 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2260 | — | — | 0.9935 | 0.5020 |
| M1_b12_hidden_parity_s42 | hidden_parity | 4127 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2180 | — | — | 0.9968 | 0.4778 |
| M1_b12_visible_full_s123 | visible_full | 4127 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 740 | — | — | 0.9305 | 0.0008 |
| M1_b12_visible_full_s42 | visible_full | 4127 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 960 | — | — | 1.0000 | 0.0000 |
| M1_b12_visible_parity_s123 | visible_parity | 4127 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2520 | — | — | 0.9774 | 0.5182 |
| M1_b12_visible_parity_s42 | visible_parity | 4127 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2600 | — | — | 0.9774 | 0.5061 |
| M1_b13_hidden_full_s123 | hidden_full | 8423 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1080 | — | — | 0.9042 | 0.0004 |
| M1_b13_hidden_full_s42 | hidden_full | 8423 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1500 | — | — | 1.0000 | 0.0004 |
| M1_b13_hidden_parity_s123 | hidden_parity | 8423 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2360 | — | — | 0.9968 | 0.4972 |
| M1_b13_hidden_parity_s42 | hidden_parity | 8423 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2420 | — | — | 0.9968 | 0.4794 |
| M1_b13_visible_full_s123 | visible_full | 8423 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1160 | — | — | 1.0000 | 0.0000 |
| M1_b13_visible_full_s42 | visible_full | 8423 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1080 | — | — | 0.9786 | 0.0004 |
| M1_b13_visible_parity_s123 | visible_parity | 8423 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2060 | — | — | 1.0000 | 0.4909 |
| M1_b13_visible_parity_s42 | visible_parity | 8423 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1900 | — | — | 0.9762 | 0.5059 |
| M1_b14_hidden_full_s123 | hidden_full | 16487 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1500 | — | — | 1.0000 | 0.0002 |
| M1_b14_hidden_full_s42 | hidden_full | 16487 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 3880 | — | — | 1.0000 | 0.0000 |
| M1_b14_hidden_parity_s123 | hidden_parity | 16487 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2660 | — | — | 0.9984 | 0.4859 |
| M1_b14_hidden_parity_s42 | hidden_parity | 16487 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 2320 | — | — | 0.9980 | 0.5021 |
| M1_b14_visible_full_s123 | visible_full | 16487 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1260 | — | — | 1.0000 | 0.0000 |
| M1_b14_visible_full_s42 | visible_full | 16487 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1520 | — | — | 1.0000 | 0.0002 |
| M1_b14_visible_parity_s123 | visible_parity | 16487 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2920 | — | — | 0.9980 | 0.4965 |
| M1_b14_visible_parity_s42 | visible_parity | 16487 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 4020 | — | — | 0.9992 | 0.5041 |
| M1_b7_hidden_full_s123 | hidden_full | 227 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 460 | — | — | 1.0000 | 0.0000 |
| M1_b7_hidden_full_s42 | hidden_full | 227 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 980 | — | — | 1.0000 | 0.0000 |
| M1_b7_hidden_parity_s123 | hidden_parity | 227 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 540 | — | — | 1.0000 | 0.3881 |
| M1_b7_hidden_parity_s42 | hidden_parity | 227 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1620 | — | — | 1.0000 | 0.4776 |
| M1_b7_visible_full_s123 | visible_full | 227 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 320 | — | — | 1.0000 | 0.0149 |
| M1_b7_visible_full_s42 | visible_full | 227 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 420 | — | — | 1.0000 | 0.0149 |
| M1_b7_visible_parity_s123 | visible_parity | 227 |  | 123 | adamw | 0.3 | 200000 | partial | 460 | — | — | 1.0000 | 0.5522 |
| M1_b7_visible_parity_s42 | visible_parity | 227 |  | 42 | adamw | 0.3 | 200000 | partial | 2340 | — | — | 1.0000 | 0.5522 |
| M1_b8_hidden_full_s123 | hidden_full | 347 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 380 | — | — | 1.0000 | 0.0000 |
| M1_b8_hidden_full_s42 | hidden_full | 347 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 560 | — | — | 1.0000 | 0.0097 |
| M1_b8_hidden_parity_s123 | hidden_parity | 347 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1280 | — | — | 1.0000 | 0.4660 |
| M1_b8_hidden_parity_s42 | hidden_parity | 347 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1600 | — | — | 1.0000 | 0.5049 |
| M1_b8_visible_full_s123 | visible_full | 347 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 540 | — | — | 1.0000 | 0.0000 |
| M1_b8_visible_full_s42 | visible_full | 347 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 680 | — | — | 1.0000 | 0.0097 |
| M1_b8_visible_parity_s123 | visible_parity | 347 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1200 | — | — | 1.0000 | 0.4272 |
| M1_b8_visible_parity_s42 | visible_parity | 347 |  | 42 | adamw | 0.3 | 200000 | partial | 1460 | — | — | 0.9615 | 0.5922 |
| M1_b9_hidden_full_s123 | hidden_full | 563 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 620 | — | — | 1.0000 | 0.0060 |
| M1_b9_hidden_full_s42 | hidden_full | 563 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 1080 | — | — | 0.6310 | 0.0000 |
| M1_b9_hidden_parity_s123 | hidden_parity | 563 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 2580 | — | — | 1.0000 | 0.5238 |
| M1_b9_hidden_parity_s42 | hidden_parity | 563 |  | 42 | adamw | 0.3 | 200000 | partial | 2040 | — | — | 0.9881 | 0.5952 |
| M1_b9_visible_full_s123 | visible_full | 563 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 440 | — | — | 1.0000 | 0.0060 |
| M1_b9_visible_full_s42 | visible_full | 563 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 700 | — | — | 1.0000 | 0.0000 |
| M1_b9_visible_parity_s123 | visible_parity | 563 |  | 123 | adamw | 0.3 | 200000 | memorization_only | 1960 | — | — | 1.0000 | 0.5000 |
| M1_b9_visible_parity_s42 | visible_parity | 563 |  | 42 | adamw | 0.3 | 200000 | memorization_only | 900 | — | — | 0.9762 | 0.5000 |


## Stage 1 vs Stage 2: Detailed Comparison

| Parameter | Stage 1 (R1_b16_s42) | Stage 2 (R2_b16_s42) | Same? |
|-----------|----------------------|----------------------|-------|
| dataset_hash | ebae03a7cb281d9f | ebae03a7cb281d9f | YES |
| p | 54721 | 54721 | YES |
| task (inferred) | DLP parity bit 0 (p=54721, b=16bit, input=(a*x)%p, target=bit_0(x)) | DLP parity bit 0 (p=54721, b=16bit, input=(a*x)%p, target=bit_0(x)) | YES |
| optimizer | adam | adamw | NO |
| weight_decay | 0 | 0.3 | NO |
| epochs | 2000 | 100000 | NO |
| phase | underfit | direct_generalization | NO |
| final_test | 0.5206666666666667 | 0.988 | NO |

**Conclusion:** Stage 1 and Stage 2 use the same task and dataset. Stage 2 is NOT modular addition.
The difference in outcome is due to optimization (WD + training duration), not group structure.


## Phase Distribution (Canonical Classification)

Using sustained thresholds (K=3 consecutive evaluations):

- Tmem = min{t : A_train(t) ≥ 0.99} (sustained)
- T90 = min{t : A_test(t) ≥ 0.90} (sustained)

| Phase | Count | Description |
|-------|-------|-------------|
| direct_generalization | 19 | T90 ≤ Tmem (generalizes before or with memorization) |
| grokking | 5 | Tmem < T90 (delayed generalization after memorization) |
| memorization_only | 137 | Tmem exists, T90 does not (memorizes but never generalizes) |
| partial | 18 | Intermediate test accuracy without reaching 90% |
| underfit | 20 | Tmem does not exist (never memorizes training set) |
