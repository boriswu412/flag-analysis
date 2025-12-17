OPENQASM 2.0;
include "qelib1.inc";

qreg q[25];          // data qubits
qreg ancX[12];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[12];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[8];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[8];      // flags measured in Z (paired with X-type stabs)


//X_type stabilizers
// Stabilizer 1: X_1 X_6 (q[0], q[5])
cx ancX[0], q[0];
cx ancX[0], q[5];

// Stabilizer 2: X_11 X_16 (q[10], q[15])
cx ancX[1], q[10];
cx ancX[1], q[15];

// Stabilizer 3: X_10 X_15 (q[9], q[14])
cx ancX[2], q[9];
cx ancX[2], q[14];

// Stabilizer 4: X_20 X_25 (q[19], q[24])
cx ancX[3], q[19];
cx ancX[3], q[24];

// Stabilizer 5: X_2 X_3 X_7 X_8 (q[1], q[2], q[6], q[7])
cx ancX[4], q[1];
cx ancX[4], flagZ[0];
cx ancX[4], q[2];
cx ancX[4], q[6];
cx ancX[4], flagZ[0];
cx ancX[4], q[7];

// Stabilizer 6: X_4 X_5 X_9 X_10 (q[3], q[4], q[8], q[9])
cx ancX[5], q[3];
cx ancX[5], flagZ[1];
cx ancX[5], q[4];
cx ancX[5], q[8];
cx ancX[5], flagZ[1];
cx ancX[5], q[9];

// Stabilizer 7: X_6 X_7 X_11 X_12 (q[5], q[6], q[10], q[11])
cx ancX[6], q[5];
cx ancX[6], flagZ[2];
cx ancX[6], q[6];
cx ancX[6], q[10];
cx ancX[6], flagZ[2];
cx ancX[6], q[11];

// Stabilizer 8: X_8 X_9 X_13 X_14 (q[7], q[8], q[12], q[13])
cx ancX[7], q[7];
cx ancX[7], flagZ[3];
cx ancX[7], q[8];
cx ancX[7], q[12];
cx ancX[7], flagZ[3];
cx ancX[7], q[13];

// Stabilizer 9: X_12 X_13 X_17 X_18 (q[11], q[12], q[16], q[17])
cx ancX[8], q[11];
cx ancX[8], flagZ[4];
cx ancX[8], q[12];
cx ancX[8], q[16];
cx ancX[8], flagZ[4];
cx ancX[8], q[17];

// Stabilizer 10: X_14 X_15 X_19 X_20 (q[13], q[14], q[18], q[19])
cx ancX[9], q[13];
cx ancX[9], flagZ[5];
cx ancX[9], q[14];
cx ancX[9], q[18];
cx ancX[9], flagZ[5];
cx ancX[9], q[19];

// Stabilizer 11: X_16 X_17 X_21 X_22 (q[15], q[16], q[20], q[21])
cx ancX[10], q[15];
cx ancX[10], flagZ[6];
cx ancX[10], q[16];
cx ancX[10], q[20];
cx ancX[10], flagZ[6];
cx ancX[10], q[21];

// Stabilizer 12: X_18 X_19 X_23 X_24 (q[17], q[18], q[22], q[23])
cx ancX[11], q[17];
cx ancX[11], flagZ[7];
cx ancX[11], q[18];
cx ancX[11], q[22];
cx ancX[11], flagZ[7];
cx ancX[11], q[23];



//Z_type stabilizers

// Stabilizer 13: Z_2 Z_3 (q[1], q[2])
cx q[1], ancZ[0];
cx q[2], ancZ[0];

// Stabilizer 14: Z_4 Z_5 (q[3], q[4])
cx q[3], ancZ[1];
cx q[4], ancZ[1];

// Stabilizer 15: Z_21 Z_22 (q[20], q[21])
cx q[20], ancZ[2];
cx q[21], ancZ[2];

// Stabilizer 16: Z_23 Z_24 (q[22], q[23])
cx q[22], ancZ[3];
cx q[23], ancZ[3];

// Stabilizer 17: Z_1 Z_2 Z_6 Z_7 (q[0], q[1], q[5], q[6])
cx q[0], ancZ[4];
cx  flagX[0], ancZ[4];
cx q[1], ancZ[4];
cx q[5], ancZ[4];
cx  flagX[0], ancZ[4];
cx q[6], ancZ[4];

// Stabilizer 18: Z_3 Z_4 Z_8 Z_9 (q[2], q[3], q[7], q[8])
cx q[2], ancZ[5];
cx  flagX[1], ancZ[5];
cx q[3], ancZ[5];
cx q[7], ancZ[5];
cx  flagX[1], ancZ[5];
cx q[8], ancZ[5];


// Stabilizer 19: Z_7 Z_8 Z_12 Z_13 (q[6], q[7], q[11], q[12])
cx q[6], ancZ[6];
cx  flagX[2], ancZ[6];
cx q[7], ancZ[6];
cx q[11], ancZ[6];
cx  flagX[2], ancZ[6];
cx q[12], ancZ[6];

// Stabilizer 20: Z_9 Z_10 Z_14 Z_15 (q[8], q[9], q[13], q[14])
cx q[8], ancZ[7];
cx  flagX[3], ancZ[7];
cx q[9], ancZ[7];
cx q[13], ancZ[7];
cx  flagX[3], ancZ[7];
cx q[14], ancZ[7];

// Stabilizer 21: Z_11 Z_12 Z_16 Z_17 (q[10], q[11], q[15], q[16])
cx q[10], ancZ[8];
cx  flagX[4], ancZ[8];
cx q[11], ancZ[8];
cx q[15], ancZ[8];
cx  flagX[4], ancZ[8];
cx q[16], ancZ[8];

// Stabilizer 22: Z_13 Z_14 Z_18 Z_19 (q[12], q[13], q[17], q[18])
cx q[12], ancZ[9];
cx  flagX[5], ancZ[9];
cx q[13], ancZ[9];
cx q[17], ancZ[9];
cx  flagX[5], ancZ[9];
cx q[18], ancZ[9];

// Stabilizer 23: Z_17 Z_18 Z_22 Z_23 (q[16], q[17], q[21], q[22])
cx q[16], ancZ[10];
cx flagX[6], ancZ[10];
cx q[17], ancZ[10];
cx q[21], ancZ[10];
cx flagX[6], ancZ[10];
cx q[22], ancZ[10];

// Stabilizer 24: Z_19 Z_20 Z_24 Z_25 (q[18], q[19], q[23], q[24])
cx q[18], ancZ[11];
cx flagX[7], ancZ[11];
cx q[19], ancZ[11];
cx q[23], ancZ[11];
cx flagX[7], ancZ[11];
cx q[24], ancZ[11];