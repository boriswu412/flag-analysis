OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancX[4];   // 4 ancilla lines (circuit 2; ancX[0..3] = combined ancX[4..7])
qreg flagZ[2];   // 2 flag lines (circuit 2; flagZ[0..1] = combined flagZ[3..4])

// --- Second circuit ---
cx ancX[0], q[10];
cx ancX[1], q[7];
cx ancX[2], q[15];
cx ancX[3], q[12];
cx ancX[0], flagZ[0];
cx ancX[3], flagZ[1];
cx ancX[1], flagZ[0];
cx ancX[0], q[11];
cx ancX[3], q[8];
cx ancX[2], flagZ[0];
cx ancX[0], q[14];
cx ancX[1], q[6];
cx ancX[3], q[13];
cx ancX[0], flagZ[0];

cx ancX[3], flagZ[1];
cx ancX[2], q[11];
cx ancX[1], q[11];
cx ancX[0], q[15];
cx ancX[3], q[9];
cx ancX[2], q[16];
cx ancX[1], flagZ[0];
cx ancX[2], flagZ[0];
cx ancX[1], q[10];
cx ancX[2], q[7];
