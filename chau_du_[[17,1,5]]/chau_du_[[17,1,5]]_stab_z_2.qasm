OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancZ[4];   // 4 ancilla lines (circuit 2; ancZ[0..3] = combined ancZ[4..7])
qreg flagX[2];  // 2 flag lines (circuit 2; flagX[0..1] = combined flagX[3..4])

// --- Second circuit ---
cx q[10], ancZ[0];
cx q[7], ancZ[1];
cx q[15], ancZ[2];
cx q[12], ancZ[3];
cx flagX[0], ancZ[0];
cx flagX[1], ancZ[3];
cx flagX[0], ancZ[1];
cx q[11], ancZ[0];
cx q[8], ancZ[3];
cx flagX[0], ancZ[2];
cx q[14], ancZ[0];
cx q[6], ancZ[1];
cx q[13], ancZ[3];
cx flagX[0], ancZ[0];

cx flagX[1], ancZ[3];
cx q[11], ancZ[2];
cx q[11], ancZ[1];
cx q[15], ancZ[0];
cx q[9], ancZ[3];
cx q[16], ancZ[2];
cx flagX[0], ancZ[1];
cx flagX[0], ancZ[2];
cx q[10], ancZ[1];
cx q[7], ancZ[2];
