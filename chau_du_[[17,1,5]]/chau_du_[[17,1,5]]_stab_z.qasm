OPENQASM 2.0;
include "qelib1.inc";

// --- Registers ---
qreg q[17];     // Data qubits
qreg ancZ[8];   // 8 ancilla lines (4 per sub-circuit)
qreg flagX[5];  // 5 flag lines (3 + 2 per sub-circuit)

//first circuit 

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


cx q[14] , ancZ[0];
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



// =============================================================================
// Second circuit — ancZ[4..7], flagX[3..4]
// =============================================================================

cx q[10], ancZ[4];
cx q[7], ancZ[5];
cx q[15], ancZ[6];
cx q[12], ancZ[7];
cx flagX[3], ancZ[4];
cx flagX[4], ancZ[7];
cx flagX[3], ancZ[5];
cx q[11], ancZ[4];
cx q[8], ancZ[7];
cx flagX[3], ancZ[6];
cx q[14], ancZ[4];
cx q[6], ancZ[5];
cx q[13], ancZ[7];
cx flagX[3], ancZ[4];

cx flagX[4], ancZ[7];
cx q[11], ancZ[6];
cx q[11], ancZ[5];
cx q[15], ancZ[4];
cx q[9], ancZ[7];
cx q[16], ancZ[6];
cx flagX[3], ancZ[5];
cx flagX[3], ancZ[6];
cx q[10], ancZ[5];
cx q[7], ancZ[6];
