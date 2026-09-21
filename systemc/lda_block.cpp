// lda_block.cpp
// A minimal SystemC model of the sEMG inference block from the NinaPro repo:
//   y = W * x + b,  W is 18 x 72,  decision = argmax(y)
// Hardware assumption: one multiply-accumulate (MAC) per clock cycle.
// The point of the model is the cycle count, not the arithmetic.

#include <systemc.h>

static const int N_IN  = 72;   // 6 time-domain features x 12 channels
static const int N_OUT = 18;   // 17 gestures + rest

SC_MODULE(LdaBlock) {
    // ---- ports: how the block talks to the outside world ----
    sc_in<bool>  clk;       // the clock; every wait() below = one tick
    sc_in<bool>  start;     // pulse high for one cycle to begin a decision
    sc_out<bool> done;      // high for one cycle when the result is ready
    sc_out<int>  argmax;    // index of the winning class

    // ---- internal storage (in real hardware: weight memory + registers) ----
    float W[N_OUT][N_IN];
    float b[N_OUT];
    float x[N_IN];          // input feature vector, filled by the testbench
    float y[N_OUT];

    unsigned long cycles = 0;   // clock ticks spent computing

    // ---- behaviour ----
    void run() {
        done.write(false);
        while (true) {
            wait();                          // wait for the next clock edge
            if (!start.read()) continue;

            for (int o = 0; o < N_OUT; ++o) {
                float acc = b[o];
                for (int i = 0; i < N_IN; ++i) {
                    acc += W[o][i] * x[i];   // one MAC ...
                    wait();                  // ... costs one cycle
                    ++cycles;
                }
                y[o] = acc;                  // one output register written
            }

            int best = 0;
            for (int o = 1; o < N_OUT; ++o) if (y[o] > y[best]) best = o;
            argmax.write(best);

            done.write(true);
            wait(); ++cycles;                // one cycle to present the result
            done.write(false);
        }
    }

    SC_CTOR(LdaBlock) {
        SC_THREAD(run);
        sensitive << clk.pos();              // run() advances on rising clock edges
    }
};

// ---- testbench: wires the block up, feeds one vector, reads the answer ----
int sc_main(int, char**) {
    sc_clock clk("clk", 12.5, SC_NS);        // 80 MHz, a Cortex-M4-class clock
    sc_signal<bool> start, done;
    sc_signal<int>  argmax;

    LdaBlock dut("lda");
    dut.clk(clk); dut.start(start); dut.done(done); dut.argmax(argmax);

    // toy weights: class k "likes" feature k, so x[5] = 1 must give class 5
    for (int o = 0; o < N_OUT; ++o) {
        dut.b[o] = 0.0f;
        for (int i = 0; i < N_IN; ++i) dut.W[o][i] = (i == o) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < N_IN; ++i) dut.x[i] = 0.0f;
    dut.x[5] = 1.0f;

    start.write(false);
    sc_start(25, SC_NS);                     // let the clock settle
    start.write(true);  sc_start(12.5, SC_NS);
    start.write(false);
    while (!done.read()) sc_start(12.5, SC_NS);

    std::cout << "class = "    << argmax.read()
              << "   cycles = " << dut.cycles
              << "   sim time = " << sc_time_stamp() << std::endl;
    // measured 2026-09-20: class = 5   cycles = 1296   sim time = 16237500 ps
    // (18*72 MACs; sc_main reads the result the moment `done` rises, before the
    //  final presentation cycle is counted)  ->  16.2 us at 80 MHz
    return 0;
}
