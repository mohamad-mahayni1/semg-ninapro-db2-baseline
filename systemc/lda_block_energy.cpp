// lda_block_energy.cpp
// Third version of the sEMG inference block (baseline: lda_block.cpp, fixed point +
// parallel MAC units: lda_block_fx4.cpp). This one adds an ENERGY SKETCH.
//
// The model counts events and prices them:
//   E_MAC       energy of one 16-bit multiply-accumulate in the datapath
//   E_FETCH_EXT energy of fetching one weight from external / off-block memory
//   E_FETCH_INT energy of fetching one weight from a small on-block buffer
//   E_LEAK      static energy per cycle (leakage, clock tree), per MAC unit
//
// THE CONSTANTS ARE RATIOS, NOT JOULES. They express one well-established fact:
// moving a word of data costs more than multiplying it, and moving it from far away
// costs much more than from near by (Horowitz, ISSCC 2014, "Computing's energy problem").
// Absolute numbers would need a real technology library and a real memory macro.
//
// Two weight-reuse schemes are compared at the same MAC_UNITS:
//   STATIONARY = 0  every product fetches its weight from external memory   (naive)
//   STATIONARY = 1  the 72 inputs are loaded once into a register file, and each weight
//                   is still fetched once, but inputs are reused across all 18 outputs

#define SC_INCLUDE_FX
#include <systemc.h>
#include <iomanip>

static const int N_IN  = 72;   // 6 time-domain features x 12 channels
static const int N_OUT = 18;   // 17 gestures + rest
#ifndef MAC_UNITS
#define MAC_UNITS 4
#endif
#ifndef STATIONARY
#define STATIONARY 0
#endif

// --- energy model, in arbitrary units (see the note above) -------------------
static const double E_MAC       = 1.0;    // one 16-bit multiply-accumulate
static const double E_FETCH_EXT = 5.0;    // one weight read from off-block memory
static const double E_FETCH_INT = 0.5;    // one read from an on-block register file
static const double E_LEAK      = 0.05;   // per cycle, per MAC unit

typedef sc_fixed<16, 8>  fx_t;
typedef sc_fixed<32, 16> acc_t;

SC_MODULE(LdaBlockEnergy) {
    sc_in<bool>  clk;
    sc_in<bool>  start;
    sc_out<bool> done;
    sc_out<int>  argmax;

    fx_t  W[N_OUT][N_IN];
    fx_t  b[N_OUT];
    fx_t  x[N_IN];
    acc_t y[N_OUT];

    unsigned long cycles = 0, macs = 0, fetch_ext = 0, fetch_int = 0;

    double energy() const {
        return macs * E_MAC
             + fetch_ext * E_FETCH_EXT
             + fetch_int * E_FETCH_INT
             + cycles * MAC_UNITS * E_LEAK;
    }

    void run() {
        done.write(false);
        while (true) {
            wait();
            if (!start.read()) continue;

#if STATIONARY
            // inputs loaded once into an on-block register file, then reused 18 times
            for (int i = 0; i < N_IN; ++i) ++fetch_ext;      // x[] read once from outside
            wait(); ++cycles;                                 // one cycle for the load burst
#endif
            for (int o = 0; o < N_OUT; ++o) {
                acc_t acc = b[o];
                for (int i = 0; i < N_IN; i += MAC_UNITS) {
                    for (int k = 0; k < MAC_UNITS && i + k < N_IN; ++k) {
                        acc += W[o][i + k] * x[i + k];
                        ++macs;
                        ++fetch_ext;                          // the weight itself: always external
#if STATIONARY
                        ++fetch_int;                          // the input: cheap, from the buffer
#else
                        ++fetch_ext;                          // the input: re-read from outside
#endif
                    }
                    wait(); ++cycles;
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

    SC_CTOR(LdaBlockEnergy) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }
};

int sc_main(int, char**) {
    sc_clock clk("clk", 12.5, SC_NS);          // 80 MHz
    sc_signal<bool> start, done;
    sc_signal<int>  argmax;

    LdaBlockEnergy dut("lda_energy");
    dut.clk(clk); dut.start(start); dut.done(done); dut.argmax(argmax);

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

    std::cout << std::fixed << std::setprecision(1)
              << "MAC_UNITS = "  << MAC_UNITS
              << "  STATIONARY = " << STATIONARY
              << "  class = "     << argmax.read()
              << "  cycles = "    << dut.cycles
              << "  macs = "      << dut.macs
              << "  fetch_ext = " << dut.fetch_ext
              << "  fetch_int = " << dut.fetch_int
              << "  energy = "    << dut.energy() << " units"
              << std::endl;
    return 0;
}
