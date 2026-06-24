OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
qreg ancX[1];
qreg ancZ[0];
qreg flagX[0];
qreg flagZ[1];


// =========================
// Stabilizer : (XIXIXIX)
// ancilla = ancX[0] (X basis), flag = flagZ[0] (Z basis)
// =========================
cx ancX[0], q[0];
cx ancX[0], flagZ[0];
cx ancX[0], q[2];
cx ancX[0], q[4];
cx ancX[0], flagZ[0];
cx ancX[0], q[6];

barrier;
