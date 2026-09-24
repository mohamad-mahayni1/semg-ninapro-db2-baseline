# SystemC models of the inference block

Three models of the classifier from this repository — the 18 x 72 shrinkage-LDA matrix-vector
multiply (72 time-domain features in, 18 classes out) — as a dedicated hardware block.

| File | What it adds |
|---|---|
| `lda_block.cpp` | baseline: `float`, one multiply-accumulate (MAC) per cycle, cycle counter |
| `lda_block_fx4.cpp` | fixed point (`sc_fixed<16,8>`, 32-bit accumulator) and `MAC_UNITS` parallel MAC units |
| `lda_block_energy.cpp` | energy sketch: a cost per MAC, per far fetch, per on-block fetch, per cycle of leakage — and three dataflows |

Measured with the Accellera SystemC 3.0 reference implementation (g++ 13.3, Ubuntu 24.04).
Every variant returns the same class on the toy weights.

## 1. Cycles

| MAC units | Cycles per decision | At 80 MHz | At 32.768 kHz | Weight reads |
|---|---|---|---|---|
| 1 | 1,296 | 16.2 us | 40 ms | 1,296 |
| 2 | 648 | 8.1 us | 20 ms | 1,296 |
| 4 | 324 | 4.1 us | 10 ms | 1,296 |
| 8 | 162 | 2.1 us | 5 ms | 1,296 |

A Cortex-M4 spends roughly the one-unit cycle count in software, since it has a single-cycle
multiply-accumulate. A block of this shape is **not a cycle win** over the microcontroller.

## 2. Energy: three dataflows

A *dataflow* is the answer to one question — what sits next to the multiplier.

| Dataflow | What is held on-block | Far reads per decision |
|---|---|---|
| none | nothing | 2,592 (every weight and every input, every time) |
| input-stationary | the 72 features, loaded once per decision | 1,368 |
| weight + input-stationary | the 72 features **and** all 1,296 weights | 72 + the weight load, amortised |

The weights never change after training, so they can be loaded once at power-up and read from
on-block memory for the rest of the device's life. `DECISIONS_PER_LOAD` spreads that one-off
load across the decisions it serves. 1,296 weights x 2 bytes = **2.6 kB**, which fits on-chip.

Costs are **ratios, not joules**: moving a word costs more than multiplying it, and moving it
from far away costs much more than from a register file beside the multiplier (Horowitz,
ISSCC 2014). Absolute numbers would need a technology library and a real memory macro.

```
E_MAC = 1    E_FETCH_EXT = 5    E_FETCH_INT = 0.5    E_LEAK = 0.05 per cycle per MAC unit
```

At four MAC units:

| Dataflow | Decisions per weight load | Far reads | On-block reads | Energy | vs naive |
|---|---|---|---|---|---|
| none | — | 2,592 | 0 | **14,321** | — |
| input-stationary | — | 1,368 | 1,296 | **8,849** | −38% |
| weight + input | 1 | 1,368 | 2,592 | **9,497** | −34% |
| weight + input | 10 | 202 | 2,592 | **3,665** | −74% |
| weight + input | 1,000 | 73 | 2,592 | **3,024** | −79% |
| weight + input | infinity | 72 | 2,592 | **3,017** | −79% |

## 3. What the tables say together

- **Parallelism buys latency, not energy.** Eight MAC units are eight times faster than one and
  cost the same 14,321 units. The multipliers were never the expensive part.
- **Data movement is the expensive part.** Far fetches are 90% of the naive energy; the 1,296
  multiplies are 9%. An energy argument that counts operations and ignores memory traffic is
  measuring the wrong thing.
- **Reuse within a decision: −38%.** The same 72 features feed all 18 classes. Load them once
  into a register file instead of re-reading them per class. Costs one cycle.
- **Reuse across decisions: −79%.** The weights are fixed after training. Loading them once and
  keeping them on-block removes the last far traffic. The floor is 3,017 units: 1,296 multiplies,
  2,592 cheap on-block reads, 72 far reads for the new features, and leakage.
- **Caching is not free, and the model says so.** With one decision per load, weight-stationary
  is *worse* than input-stationary (9,497 vs 8,849): you pay 1,296 far reads to fill the buffer
  and then pay again to read it. Break-even is at **two decisions**; everything after that is
  profit. A real implant makes millions, so the load cost vanishes.
- **Leakage is noise here** (65 units, 0.5%) because the block is tiny and busy while it runs.
  In a real implant, idle most of the time, static power and how long the core sleeps between
  decisions would dominate this whole table.
- **This is the cheapest stage of the chain.** The classifier runs once per 200 ms window;
  filtering and feature extraction run at the 2 kHz sample rate. The stage worth a dedicated
  block is decided by the front end, not by this multiply.

**The generalisation.** Whether the weights can stay on-block is decided by model size, not by
cleverness. Below the on-chip memory limit the weights stop travelling and the −79% is
available; above it they stream from far memory on every decision and no dataflow can help.
That is the energy argument for model compression — not "a small model uses less memory", but
"below a threshold the weights stop moving".

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
g++ -std=c++17 -O2 -DMAC_UNITS=4 -DDATAFLOW=2 -DDECISIONS_PER_LOAD=1000 \
    -I$SC/include -L$SC/lib lda_block_energy.cpp -lsystemc -o lda_energy
LD_LIBRARY_PATH=$SC/lib ./lda_energy
```

`MAC_UNITS` sets the parallel multipliers (1, 2, 4, 8). `DATAFLOW` selects 0 = none,
1 = input-stationary, 2 = weight + input-stationary. `DECISIONS_PER_LOAD` amortises the
one-off weight load.

## Honest limits

- The energy constants are plausible ratios, not measurements. No claim is made about joules.
- No place-and-route, no real memory macro, no clock-gating model.
- The toy weight matrix exercises the datapath, not the classifier's accuracy — that lives in
  `REPORT.md` and `src/`.
