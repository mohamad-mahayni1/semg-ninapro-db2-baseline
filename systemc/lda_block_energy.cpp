// lda_block_energy.cpp
// Third version of the sEMG inference block (baseline: lda_block.cpp, fixed point +
// parallel MAC units: lda_block_fx4.cpp). This one adds an ENERGY SKETCH and compares
// three dataflows -- three answers to "what sits next to the multiplier".
//
//   DATAFLOW = 0   nothing held on-block. Every product fetches both its weight and its
//                  input from far memory. 2,592 far reads per decision.
//   DATAFLOW = 1   input-stationary. The 72 features are loaded once into an on-block
//                  register file and read from there 18 times. Weights still come from far
//                  memory, because within one decision each weight is used exactly once.
//   DATAFLOW = 2   weight- and input-stationary. The 1,296 weights never change after
//                  training, so they are loaded once at power-up (amortised over
//                  DECISIONS_PER_LOAD decisions) and read from on-block memory thereafter.
//                  1,296 weights x 2 bytes = 2.6 kB, which fits on-chip.
//
// The constants are RATIOS, NOT JOULES. They encode one established fact: moving a word
// costs more than multiplying it, and moving it from far away costs much more than from a
// register file beside the multiplier (Horowitz, ISSCC 2014). Absolute numbers would need a
// technology library and a real memory macro.

#define SC_INCLUDE_FX
#include <systemc.h>
#include <iomanip>

static const int N_IN  = 72;   // 6 time-domain features x 12 channels
static const int N_OUT = 18;   // 17 gestures + rest
#ifndef MAC_UNITS
#define MAC_UNITS 4
#endif
#ifndef DATAFLOW
#define DATAFLOW 0
#endif
// How many decisions one weight load serves. The weights are fixed after training, so in a
// real device this is "every decision until the battery dies"; 1 shows the worst case.
#ifndef DECISIONS_PER_LOAD
#define DECISIONS_PER_LOAD 1
#endif

// --- energy model, in arbitrary units (see the note above) -------------------
static const double E_MAC       = 1.0;    // one 16-bit multiply-accumulate
static const double E_FETCH_EXT = 5.0;    // one read from off-block memory
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

    unsigned long cycles = 0, macs = 0, fetch_int = 0;
    double fetch_ext = 0.0;   // fractional: a one-off weight load is spread over decisions

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

#if DATAFLOW == 2
            // one-off weight load into on-block memory, charged pro rata to this decision
            fetch_ext += double(N_OUT * N_IN) / DECISIONS_PER_LOAD;
            wait(); ++cycles;                         // one cycle for the load burst
#endif
#if DATAFLOW >= 1
            for (int i = 0; i < N_IN; ++i) fetch_ext += 1.0;   // the 72 features, once
            wait(); ++cycles;
#endif
            for (int o = 0; o < N_OUT; ++o) {
                acc_t acc = b[o];
                for (int i = 0; i < N_IN; i += MAC_UNITS) {
                    for (int k = 0; k < MAC_UNITS && i + k < N_IN; ++k) {
                        acc += W[o][i + k] * x[i + k];
                        ++macs;
#if DATAFLOW == 2
                        ++fetch_int;                  // weight: on-block
                        ++fetch_int;                  // input:  on-block
#elif DATAFLOW == 1
                        fetch_ext += 1.0;             // weight: far
                        ++fetch_int;                  // input:  on-block
#else
                        fetch_ext += 1.0;             // weight: far
                        fetch_ext += 1.0;             // input:  far
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

    const char* name[3] = {"none", "input-stationary", "weight+input-stationary"};
    std::cout << std::fixed << std::setprecision(1)
              << "dataflow = " << name[DATAFLOW]
              << "  mac_units = "  << MAC_UNITS
              << "  loads/1 = "    << DECISIONS_PER_LOAD
              << "  class = "      << argmax.read()
              << "  cycles = "     << dut.cycles
              << "  macs = "       << dut.macs
              << "  far = "        << dut.fetch_ext
              << "  near = "       << dut.fetch_int
              << "  energy = "     << dut.energy()
              << std::endl;
    return 0;
}
