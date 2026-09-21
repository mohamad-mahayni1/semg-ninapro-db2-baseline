# SystemC model of the inference block

`lda_block.cpp` models the classifier from this repository — the 18 x 72 shrinkage-LDA
matrix-vector multiply (72 time-domain features in, 18 classes out) — as a hardware block
that performs one multiply-accumulate (MAC) per clock cycle, with a cycle counter.

Measured with the Accellera SystemC 3.0 reference implementation (g++ 13.3, Ubuntu 24.04):

```
class = 5   cycles = 1296   sim time = 16237500 ps
```

18 x 72 = 1,296 MACs per decision. At 80 MHz that is 16.2 us; on a 32.768 kHz clock it is
about 40 ms. A Cortex-M4 executes roughly the same number of MACs per cycle in software, so
the block is not a cycle win over the microcontroller. The case for a dedicated block is
energy per operation and how long the core can stay asleep — and the classifier is the
cheapest stage of the chain, since it runs once per 200 ms window while filtering and feature
extraction run at the 2 kHz sample rate.

## Build and run

```bash
# SystemC once (installs into ~/systemc-install, nothing system-wide)
git clone --depth 1 https://github.com/accellera-official/systemc.git
cd systemc && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/systemc-install -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
make -j4 && make install

# this model
g++ -std=c++17 -O2 -I$HOME/systemc-install/include -L$HOME/systemc-install/lib lda_block.cpp -lsystemc -o lda_block
LD_LIBRARY_PATH=$HOME/systemc-install/lib ./lda_block
```

## Next steps

- **Parallel MAC units:** four MACs per cycle (unroll the inner loop by four, one `wait()`
  per four products) — about 325 cycles. More multipliers, fewer cycles, more silicon.
- **Fixed point:** `sc_fixed<16,8>` instead of `float` — the step from "fits in memory" to
  "runs on a microcontroller-class block".
- **Energy sketch:** a cost per MAC and a cost per weight fetch, so the model shows where the
  energy goes once fetches cost more than multiplies.
