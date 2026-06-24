OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
qreg ancX[1];

// Stabilizer : (IIIXXXX) — raw (no flags)
cx ancX[0], q[3];
cx ancX[0], q[4];
cx ancX[0], q[5];
cx ancX[0], q[6];

barrier;
