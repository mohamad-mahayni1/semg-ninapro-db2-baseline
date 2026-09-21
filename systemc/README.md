# SystemC models of the inference block

Two models of the classifier from this repository — the 18 x 72 shrinkage-LDA matrix-vector
multiply (72 time-domain features in, 18 classes out) — as a dedicated hardware block.

| File | Arithmetic | MAC units per cycle | Cycles per decision | At 80 MHz |
|---|---|---|---|---|
| `lda_block.cpp` | `float` | 1 | **1,296** | 16.2 us |
| `lda_block_fx4.cpp` | `sc_fixed<16,8>`, `sc_fixed<32,16>` accumulator | 1 | 1,296 | 16.2 us |
| `lda_block_fx4.cpp -DMAC_UNITS=2` | fixed point | 2 | 648 | 8.1 us |
| `lda_block_fx4.cpp` (default) | fixed point | **4** | **324** | 4.1 us |
| `lda_block_fx4.cpp -DMAC_UNITS=8` | fixed point | 8 | 162 | 2.1 us |

Measured with the Accellera SystemC 3.0 reference implementation (g++ 13.3, Ubuntu 24.04).
All variants return the same class on the toy weights.

## What the sweep says

- **Cycles scale down with parallel MAC units; weight fetches do not.** Every row fetches
  1,296 weights per decision. With four units the block is four times faster and reads
  exactly as much memory — so if a weight fetch costs more than a multiply (it usually does),
  the energy per decision barely moves. Parallelism buys latency, not energy.
- **A Cortex-M4 spends roughly the 1,296 cycles of the one-unit block in software**, since it
  has a single-cycle multiply-accumulate. The case for a dedicated block is therefore energy
  per operation and how long the core can sleep, not cycle count.
- **The classifier is the cheapest stage of the chain.** It runs once per 200 ms window;
  filtering and feature extraction run at the 2 kHz sample rate. The stage worth a block is
  decided by the front end, not by this multiply.
- On a 32.768 kHz clock the one-unit block takes about 40 ms per decision; the four-unit block
  about 10 ms.

## Build and run

```bash
# SystemC once (installs into ~/systemc-install, nothing system-wide)
git clone --depth 1 https://github.com/accellera-official/systemc.git
cd systemc && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/systemc-install -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
make -j4 && make install

# baseline
g++ -std=c++17 -O2 -I$HOME/systemc-install/include -L$HOME/systemc-install/lib lda_block.cpp -lsystemc -o lda_block
LD_LIBRARY_PATH=$HOME/systemc-install/lib ./lda_block

# fixed point, N parallel MAC units
g++ -std=c++17 -O2 -DMAC_UNITS=4 -I$HOME/systemc-install/include -L$HOME/systemc-install/lib lda_block_fx4.cpp -lsystemc -o lda_fx
LD_LIBRARY_PATH=$HOME/systemc-install/lib ./lda_fx
```

## Next step

An energy sketch: a cost per MAC and a cost per weight fetch, so the table above turns into
energy per decision and the fetch-versus-multiply point becomes a number.
