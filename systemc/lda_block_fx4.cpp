// lda_block_fx4.cpp
// Second version of the sEMG inference block (see lda_block.cpp for the baseline):
//   * fixed-point arithmetic  : sc_fixed<16,8> weights and inputs (8 integer bits, 8 fraction
//                               bits), sc_fixed<32,16> accumulator  -> no floating-point unit
//   * parallel MAC units      : MAC_UNITS multiply-accumulates per clock cycle (default 4)
// Cycles per decision drop from 18*72 = 1296 to 18*ceil(72/MAC_UNITS) = 324 at MAC_UNITS = 4.
// The cost is MAC_UNITS multipliers instead of one, and MAC_UNITS weight fetches per cycle.

#define SC_INCLUDE_FX
#include <systemc.h>

static const int N_IN  = 72;   // 6 time-domain features x 12 channels
static const int N_OUT = 18;   // 17 gestures + rest
#ifndef MAC_UNITS
#define MAC_UNITS 4                 // parallel multiply-accumulate units
#endif

typedef sc_fixed<16, 8>  fx_t;   // data path word: 16 bits, 8 before the point
typedef sc_fixed<32, 16> acc_t;  // accumulator: wide enough for 72 products

SC_MODULE(LdaBlockFx) {
    sc_in<bool>  clk;
    sc_in<bool>  start;
    sc_out<bool> done;
    sc_out<int>  argmax;

    fx_t  W[N_OUT][N_IN];
    fx_t  b[N_OUT];
    fx_t  x[N_IN];
    acc_t y[N_OUT];

    unsigned long cycles       = 0;   // clock ticks spent computing
    unsigned long weight_reads = 0;   // one per weight fetched from memory
    unsigned long macs         = 0;   // multiply-accumulates performed

    void run() {
        done.write(false);
        while (true) {
            wait();
            if (!start.read()) continue;

            for (int o = 0; o < N_OUT; ++o) {
                acc_t acc = b[o];
                for (int i = 0; i < N_IN; i += MAC_UNITS) {
                    // MAC_UNITS products computed in the same cycle, each on its own multiplier
                    for (int k = 0; k < MAC_UNITS && i + k < N_IN; ++k) {
                        acc += W[o][i + k] * x[i + k];
                        ++macs; ++weight_reads;
                    }
                    wait(); ++cycles;                // one cycle for the whole group
                }
                y[o] = acc;
            }

            int best = 0;
            for (int o = 1; o < N_OUT; ++o) if (y[o] > y[best]) best = o;
            argmax.write(best);

            done.write(true);
            wait(); ++cycles;
            done.write(false);
        }
    }

    SC_CTOR(LdaBlockFx) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }
};

int sc_main(int, char**) {
    sc_clock clk("clk", 12.5, SC_NS);          // 80 MHz
    sc_signal<bool> start, done;
    sc_signal<int>  argmax;

    LdaBlockFx dut("lda_fx");
    dut.clk(clk); dut.start(start); dut.done(done); dut.argmax(argmax);

    // toy weights, same as the baseline: class k prefers feature k
    for (int o = 0; o < N_OUT; ++o) {
        dut.b[o] = 0;
        for (int i = 0; i < N_IN; ++i) dut.W[o][i] = (i == o) ? 1.0 : 0.0;
    }
    for (int i = 0; i < N_IN; ++i) dut.x[i] = 0;
    dut.x[5] = 1.0;

    start.write(false);
    sc_start(25, SC_NS);
    start.write(true);  sc_start(12.5, SC_NS);
    start.write(false);
    while (!done.read()) sc_start(12.5, SC_NS);

    std::cout << "MAC_UNITS = " << MAC_UNITS
              << "   class = "        << argmax.read()
              << "   cycles = "       << dut.cycles
              << "   macs = "         << dut.macs
              << "   weight_reads = " << dut.weight_reads
              << "   sim time = "     << sc_time_stamp() << std::endl;
    return 0;
}
