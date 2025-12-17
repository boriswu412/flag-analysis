OPENQASM 2.0;
include "qelib1.inc";

qreg q[17];          // data qubits
qreg ancX[8];        // ancillas measured in X (for X-type stabilizers)
qreg ancZ[8];        // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[11];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[11];      // flags measured in Z (paired with X-type stabs)


// For the Z-type stabilizers with flags

// Row 1: 111100000000000  -> q[0], q[1], q[2], q[3]
cx q[0], ancZ[0];
cx flagX[0], ancZ[0];
cx q[1], ancZ[0];
cx q[2], ancZ[0];
cx flagX[0], ancZ[0];
cx q[3], ancZ[0];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 2: 1010110000000000  -> q[0], q[2], q[4], q[5]
cx q[0], ancZ[1];
cx flagX[1], ancZ[1];
cx q[2], ancZ[1];
cx q[4], ancZ[1];
cx flagX[1], ancZ[1];
cx q[5], ancZ[1];


// Row 3: 00001100110000000  -> q[4], q[5], q[8], q[9]
cx q[4], ancZ[2];
cx flagX[2], ancZ[2];
cx q[5], ancZ[2];
cx q[8], ancZ[2];
cx flagX[2], ancZ[2];
cx q[9], ancZ[2];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 4: 00000011001100000  -> q[6], q[7], q[10], q[11]
cx q[6], ancZ[3];
cx flagX[3], ancZ[3];
cx q[7], ancZ[3];
cx q[10], ancZ[3];
cx flagX[3], ancZ[3];
cx q[11], ancZ[3];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 5: 00000000110011000  -> q[8], q[9], q[12], q[13]
cx q[8], ancZ[4];
cx flagX[4], ancZ[4];
cx q[9], ancZ[4];
cx q[12], ancZ[4];
cx flagX[4], ancZ[4];
cx q[13], ancZ[4];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 6: 00000000001100110  -> q[10], q[11], q[14], q[15]
cx q[10], ancZ[5];
cx flagX[5], ancZ[5];
cx q[11], ancZ[5];
cx q[14], ancZ[5];
cx flagX[5], ancZ[5];
cx q[15], ancZ[5];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 7: 00000001000100011  -> q[7], q[11], q[15], q[16]
cx q[7], ancZ[6];
cx flagX[6], ancZ[6];
cx q[11], ancZ[6];
cx q[15], ancZ[6];
cx flagX[6], ancZ[6];
cx q[16], ancZ[6];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 8: 00110110011001100  -> q[2], q[3], q[5], q[6], q[9], q[10], q[13], q[14]

cx q[2], ancZ[7];
cx flagX[7], ancZ[7];
cx q[3], ancZ[7];
cx flagX[8], ancZ[7];
cx q[5], ancZ[7];
cx flagX[9], ancZ[7];
cx q[6], ancZ[7];
cx flagX[10], ancZ[7];
cx flagX[7], ancZ[7];
cx q[9], ancZ[7];
cx flagX[9], ancZ[7];
cx q[10], ancZ[7];
cx q[13], ancZ[7];
cx flagX[8], ancZ[7];
cx flagX[10], ancZ[7];
cx q[14], ancZ[7];

//For the X-type stabilizers with flags

// Row 1: 111100000000000  -> q[0], q[1], q[2], q[3]
cx ancX[0], q[0];
cx ancX[0], flagZ[0];
cx ancX[0], q[1];
cx ancX[0], q[2];
cx ancX[0], flagZ[0];
cx ancX[0], q[3];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 2: 1010110000000000  -> q[0], q[2], q[4], q[5]
cx ancX[1], q[0];
cx ancX[1], flagZ[1];
cx ancX[1], q[2];
cx ancX[1], q[4];
cx ancX[1], flagZ[1];
cx ancX[1], q[5];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 3: 00001100110000000  -> q[4], q[5], q[8], q[9]
cx ancX[2], q[4];
cx ancX[2], flagZ[2];
cx ancX[2], q[5];
cx ancX[2], q[8];
cx ancX[2], flagZ[2];
cx ancX[2], q[9];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 4: 00000011001100000  -> q[6], q[7], q[10], q[11]
cx ancX[3], q[6];
cx ancX[3], flagZ[3];
cx ancX[3], q[7];
cx ancX[3], q[10];
cx ancX[3], flagZ[3];
cx ancX[3], q[11];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 5: 00000000110011000  -> q[8], q[9], q[12], q[13]
cx ancX[4], q[8];
cx ancX[4], flagZ[4];
cx ancX[4], q[9];
cx ancX[4], q[12];
cx ancX[4], flagZ[4];
cx ancX[4], q[13];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 6: 00000000001100110  -> q[10], q[11], q[14], q[15]
cx ancX[5], q[10];
cx ancX[5], flagZ[5];
cx ancX[5], q[11];
cx ancX[5], q[14];
cx ancX[5], flagZ[5];
cx ancX[5], q[15];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 7: 00000001000100011  -> q[7], q[11], q[15], q[16]
cx ancX[6], q[7];
cx ancX[6], flagZ[6];
cx ancX[6], q[11];
cx ancX[6], q[15];
cx ancX[6], flagZ[6];
cx ancX[6], q[16];

barrier q, ancX, ancZ, flagX, flagZ;

// Row 8: 00110110011001100
cx ancX[7], q[2];
cx ancX[7], flagZ[7];
cx ancX[7], q[3];
cx ancX[7], flagZ[8];
cx ancX[7], q[5];
cx ancX[7], flagZ[9];
cx ancX[7], q[6];
cx ancX[7], flagZ[10];
cx ancX[7], flagZ[7];
cx ancX[7], q[9];
cx ancX[7], flagZ[9];
cx ancX[7], q[10];
cx ancX[7], q[13];
cx ancX[7], flagZ[8];
cx ancX[7], flagZ[10];
cx ancX[7], q[14];

barrier q, ancX, ancZ, flagX, flagZ;