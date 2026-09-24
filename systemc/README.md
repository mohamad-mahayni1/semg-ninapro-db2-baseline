# SystemC models of the inference block

Three models of the classifier from this repository — the 18 x 72 shrinkage-LDA matrix-vector
multiply (72 time-domain features in, 18 classes out) — as a dedicated hardware block.

| File | What it adds |
|---|---|
| `lda_block.cpp` | baseline: `float`, one multiply-accumulate (MAC) per cycle, cycle counter |
| `lda_block_fx4.cpp` | fixed point (`sc_fixed<16,8>`, 32-bit accumulator) and `MAC_UNITS` parallel MAC units |
| `lda_block_energy.cpp` | an energy sketch: a cost per MAC, per external fetch, per on-block fetch, per cycle of leakage — and a weight/input reuse scheme |

Measured with the Accellera SystemC 3.0 reference implementation (g++ 13.3, Ubuntu 24.04).
Every variant returns the same class on the toy weights.

## 1. Cycles

| MAC units | Cycles per decision | At 80 MHz | At 32.768 kHz | Weight fetches |
|---|---|---|---|---|
| 1 | 1,296 | 16.2 us | 40 ms | 1,296 |
| 2 | 648 | 8.1 us | 20 ms | 1,296 |
| 4 | 324 | 4.1 us | 10 ms | 1,296 |
| 8 | 162 | 2.1 us | 5 ms | 1,296 |

A Cortex-M4 spends roughly the one-unit cycle count in software, since it has a single-cycle
multiply-accumulate. So a block of this shape is **not a cycle win** over the microcontroller.

## 2. Energy

Same block, priced. The constants are **ratios, not joules**: a data movement costs more than
an arithmetic operation, and a move from off-block memory costs much more than one from a
register file next to the multiplier (Horowitz, ISSCC 2014). Absolute numbers would need a
technology library and a real memory macro.

```
E_MAC = 1    E_FETCH_EXT = 5    E_FETCH_INT = 0.5    E_LEAK = 0.05 per cycle per MAC unit
```

| MAC units | Reuse | Cycles | MACs | External fetches | On-block fetches | Energy |
|---|---|---|---|---|---|---|
| 1 | none | 1,296 | 1,296 | 2,592 | 0 | **14,321** |
| 4 | none | 324 | 1,296 | 2,592 | 0 | **14,321** |
| 8 | none | 162 | 1,296 | 2,592 | 0 | **14,321** |
| 1 | inputs held on-block | 1,297 | 1,296 | 1,368 | 1,296 | **8,849** |
| 4 | inputs held on-block | 325 | 1,296 | 1,368 | 1,296 | **8,849** |
| 8 | inputs held on-block | 163 | 1,296 | 1,368 | 1,296 | **8,849** |

## 3. What the two tables say together

- **Parallelism buys latency, not energy.** Eight MAC units are eight times faster than one
  and cost the same 14,321 units. The multipliers were never the expensive part.
- **Data movement is the expensive part.** Fetches are 90% of the naive energy; the 1,296
  multiplies are 9%. Any energy argument that counts operations and ignores memory traffic is
  measuring the wrong thing.
- **Reuse is the lever.** Loading the 72 inputs once into an on-block register file and reading
  them from there, instead of re-reading each input for every output, cuts energy by **38%**
  and changes the cycle count by one. Latency and energy are two different design knobs.
- **Leakage is noise here** (65 units, 0.5%) because the block is tiny and always busy while
  it runs. In a real implant, where the block is idle most of the time, static power and how
  long the core sleeps between decisions dominate this whole table.
- **This is the cheapest stage of the chain.** The classifier runs once per 200 ms window;
  filtering and feature extraction run at the 2 kHz sample rate. The stage worth a dedicated
  block is decided by the front end, not by this multiply.

## Build and run

```bash
# SystemC once (installs into ~/systemc-install, nothing system-wide)
git clone --depth 1 https://github.com/accellera-official/systemc.git
cd systemc && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/systemc-install -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
make -j4 && make install

SC=$HOME/systemc-install
g++ -std=c++17 -O2 -I$SC/include -L$SC/lib lda_block.cpp -lsystemc -o lda_block
g++ -std=c++17 -O2 -DMAC_UNITS=4 -I$SC/include -L$SC/lib lda_block_fx4.cpp -lsystemc -o lda_fx
g++ -std=c++17 -O2 -DMAC_UNITS=4 -DSTATIONARY=1 -I$SC/include -L$SC/lib lda_block_energy.cpp -lsystemc -o lda_energy
LD_LIBRARY_PATH=$SC/lib ./lda_energy
```

`MAC_UNITS` sets the parallel multipliers (1, 2, 4, 8). `STATIONARY=1` holds the input vector
in an on-block register file instead of re-reading it for every output.

## Honest limits

- The energy constants are plausible ratios, not measurements. No claim is made about joules.
- No place-and-route, no real memory macro, no clock-gating model.
- The toy weight matrix exercises the datapath, not the classifier's accuracy — that lives in
  `REPORT.md` and `src/`.
