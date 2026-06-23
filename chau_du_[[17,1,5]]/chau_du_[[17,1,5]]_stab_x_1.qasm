OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancX[4];   // 4 ancilla lines (circuit 1)
qreg flagZ[3];   // 3 flag lines (circuit 1)

// --- Group 1 ---
cx ancX[0], q[2]; // Pink, line 3
cx ancX[2], q[8]; // Red, line 9
cx ancX[3], q[5]; // Green, line 6
cx ancX[1], q[2]; // Blue, line 3

cx ancX[2], flagZ[1]; // Black, top flag
cx ancX[0], flagZ[0];
cx ancX[3], flagZ[2]; // Black, bottom flag
cx ancX[1], flagZ[0];

// --- Group 2 ---
cx ancX[0], q[3]; // Pink, line 4
cx ancX[0], flagZ[1]; // Black, middle flag
cx ancX[2], q[9]; // Red, line 10
cx ancX[3], q[2]; // Green, line 3
cx ancX[1], q[3]; // Blue, line 4

cx ancX[2], q[5];
cx ancX[3], q[0];

cx ancX[0], q[14];
cx ancX[1], q[1];
cx ancX[2], flagZ[1]; // Black, middle flag
cx ancX[3], flagZ[2];

// --- Group 3 ---
cx ancX[1], flagZ[0];
cx ancX[0], q[13];
cx ancX[2], q[4]; // Red, line 6
cx ancX[3], q[4]; // Green, line 1

cx ancX[0], flagZ[2]; // Black, bottom flag
cx ancX[1], q[0];

cx ancX[0], q[10];
cx ancX[0], q[6];
cx ancX[0], flagZ[0];
cx ancX[0], q[9];
cx ancX[0], flagZ[2];
cx ancX[0], flagZ[1];
cx ancX[0], q[5];
