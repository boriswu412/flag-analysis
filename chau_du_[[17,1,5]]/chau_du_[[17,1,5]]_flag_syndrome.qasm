OPENQASM 2.0;
include "qelib1.inc";

qreg q[17];
qreg ancX[8];   // X stabilizer ancillas
qreg flagZ[5];   // flags paired with X stabilizers
qreg ancZ[8];   // Z stabilizer ancillas
qreg flagX[5];   // flags paired with Z stabilizers

// =============================================================================
// X stabilizers (circuit 1: ancX[0..3], flagZ[0..2]; circuit 2: ancX[4..7], flagZ[3..4])
// =============================================================================

cx ancX[0], q[2]; // Pink, line 3
cx ancX[2], q[8]; // Red, line 9
cx ancX[3], q[5]; // Green, line 6
cx ancX[1], q[2]; // Blue, line 3
cx ancX[2], flagZ[1]; // Black, top flag
cx ancX[0], flagZ[0];
cx ancX[3], flagZ[2]; // Black, bottom flag
cx ancX[1], flagZ[0];
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

// =============================================================================
// Z stabilizers (circuit 1: ancZ[0..3], flagX[0..2]; circuit 2: ancZ[4..7], flagX[3..4])
// =============================================================================

cx q[2], ancZ[0];     // Pink, line 3
cx q[8], ancZ[2];     // Red, line 9
cx q[5], ancZ[3];     // Green, line 6
cx q[2], ancZ[1];     // Blue, line 3
cx flagX[1], ancZ[2]; // Black, top flag
cx flagX[0], ancZ[0];
cx flagX[2], ancZ[3]; // Black, bottom flag
cx flagX[0], ancZ[1];
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
