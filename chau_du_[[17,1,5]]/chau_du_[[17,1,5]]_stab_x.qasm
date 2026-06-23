OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancX[8];   // 8 ancilla lines (4 per sub-circuit)
qreg flagZ[5];   // 5 flag lines (3 + 2 per sub-circuit)

// X stabilizer extraction (symmetric to stab_z) 

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



// =============================================================================
// Second circuit — ancX[4..7], flagZ[3..4]
// =============================================================================

cx ancX[4], q[10];
cx ancX[5], q[7];
cx ancX[6], q[15];
cx ancX[7], q[12];
cx ancX[4], flagZ[3];
cx ancX[7], flagZ[4];
cx ancX[5], flagZ[3];
cx ancX[4], q[11];
cx ancX[7], q[8];
cx ancX[6], flagZ[3];
cx ancX[4], q[14];
cx ancX[5], q[6];
cx ancX[7], q[13];
cx ancX[4], flagZ[3];

cx ancX[7], flagZ[4];
cx ancX[6], q[11];
cx ancX[5], q[11];
cx ancX[4], q[15];
cx ancX[7], q[9];
cx ancX[6], q[16];
cx ancX[5], flagZ[3];
cx ancX[6], flagZ[3];
cx ancX[5], q[10];
cx ancX[6], q[7];
