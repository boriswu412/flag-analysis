OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
qreg ancX[1];

// Stabilizer : (IXXIIXX) — raw (no flags)
cx ancX[0], q[1];
cx ancX[0], q[2];
cx ancX[0], q[5];
cx ancX[0], q[6];

barrier;
