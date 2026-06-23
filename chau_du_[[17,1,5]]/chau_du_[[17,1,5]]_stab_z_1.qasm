OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancZ[4];   // 4 ancilla lines (circuit 1)
qreg flagX[3];  // 3 flag lines (circuit 1)

// --- Group 1 ---
cx q[2], ancZ[0];     // Pink, line 3
cx q[8], ancZ[2];     // Red, line 9
cx q[5], ancZ[3];     // Green, line 6
cx q[2], ancZ[1];     // Blue, line 3

cx flagX[1], ancZ[2]; // Black, top flag
cx flagX[0], ancZ[0];
cx flagX[2], ancZ[3]; // Black, bottom flag
cx flagX[0], ancZ[1];

// --- Group 2 ---
cx q[3], ancZ[0];     // Pink, line 4
cx flagX[1], ancZ[0]; // Black, middle flag
cx q[9], ancZ[2];     // Red, line 10
cx q[2], ancZ[3];     // Green, line 3
cx q[3], ancZ[1];     // Blue, line 4

cx q[5], ancZ[2];
cx q[0], ancZ[3];

cx q[14], ancZ[0];
cx q[1], ancZ[1];
cx flagX[1], ancZ[2]; // Black, middle flag
cx flagX[2], ancZ[3];

// --- Group 3 ---
cx flagX[0], ancZ[1];
cx q[13], ancZ[0];
cx q[4], ancZ[2];     // Red, line 6
cx q[4], ancZ[3];     // Green, line 1

cx flagX[2], ancZ[0]; // Black, bottom flag
cx q[0], ancZ[1];

cx q[10], ancZ[0];
cx q[6], ancZ[0];
cx flagX[0], ancZ[0];
cx q[9], ancZ[0];
cx flagX[2], ancZ[0];
cx flagX[1], ancZ[0];
cx q[5], ancZ[0];
