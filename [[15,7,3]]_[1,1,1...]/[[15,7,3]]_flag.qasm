OPENQASM 2.0;
include "qelib1.inc";

qreg q[15];          // data qubits
qreg ancX[4];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[4];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[4];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[4];      // flags measured in Z (paired with X-type stabs)


barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 1 : 
// ancilla = ancX[0] (X basis), flag = flagZ[0] (Z basis)
// =========================
cx  ancX[0], q[14];
cx  ancX[0], flagZ[0];
cx  ancX[0], q[12];
cx  ancX[0], q[13];
cx  ancX[0], q[10];
cx  ancX[0], q[11];
cx  ancX[0], q[9];
cx  ancX[0], q[8];
cx  ancX[0], flagZ[0];
cx  ancX[0], q[7];


barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 2 : 
// ancilla = ancX[1] (X basis), flag = flagZ[2] (Z basis)
// =========================
cx  ancX[1], q[14];
cx  ancX[1], flagZ[2];
cx  ancX[1], q[12];
cx  ancX[1], q[13];
cx  ancX[1], q[6];
cx  ancX[1], q[11];
cx  ancX[1], q[5];
cx  ancX[1], q[4];
cx  ancX[1], flagZ[1];
cx  ancX[1], q[3];


barrier q, ancX, ancZ, flagX, flagZ;
// =========================
// Stabilizer 3 : 
// ancilla = ancX[2] (X basis), flag = flagZ[2] (Z basis)
// =========================
cx  ancX[2], q[14];
cx  ancX[2], flagZ[2];
cx  ancX[2], q[10];
cx  ancX[2], q[13];
cx  ancX[2], q[6];
cx  ancX[2], q[9];
cx  ancX[2], q[5];
cx  ancX[2], q[2];
cx  ancX[2], flagZ[2];
cx  ancX[2], q[1];


barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 4 : 
// ancilla = ancX[3] (X basis), flag = flagZ[3] (Z basis)
// =========================
cx  ancX[3], q[14];
cx  ancX[3], flagZ[3];
cx  ancX[3], q[10];
cx  ancX[3], q[12];
cx  ancX[3], q[6];
cx  ancX[3], q[8];
cx  ancX[3], q[4];
cx  ancX[3], q[2];
cx  ancX[3], flagZ[3];
cx  ancX[3], q[0];


barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 5 (Z syndrome):
// ancilla = ancZ[0] (Z basis), flag = flagX[0] (X basis)
// =========================
cx  q[14], ancZ[0];
cx  flagX[0], ancZ[0];
cx  q[12], ancZ[0];

cx  q[13], ancZ[0];
cx  q[10], ancZ[0];
cx  q[11], ancZ[0];
cx  q[9],  ancZ[0];
cx  q[8],  ancZ[0];
cx  flagX[0], ancZ[0];
cx  q[7],  ancZ[0];



barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 6 (Z syndrome):
// ancilla = ancZ[1] (Z basis), flag = flagX[1] (X basis)
// =========================
cx  q[14], ancZ[1];
cx  flagX[1], ancZ[1];
cx  q[12], ancZ[1];
cx  q[13], ancZ[1];
cx  q[6], ancZ[1];
cx  q[11],  ancZ[1];
cx  q[5],  ancZ[1];
cx  q[4],  ancZ[1];
cx  flagX[1], ancZ[1];
cx  q[3],  ancZ[1];



barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 7 (Z syndrome):
// ancilla = ancZ[2] (Z basis), flag = flagX[2] (X basis)
// =========================
cx  q[14], ancZ[2];
cx  flagX[2], ancZ[2];
cx  q[10], ancZ[2];
cx  q[13], ancZ[2];
cx  q[6],  ancZ[2];
cx  q[9],  ancZ[2];
cx  q[5],  ancZ[2];
cx  q[2],  ancZ[2];
cx  flagX[2], ancZ[2];
cx  q[1],  ancZ[2];



barrier q, ancX, ancZ, flagX, flagZ;


// =========================
// Stabilizer 8 (Z syndrome):
// ancilla = ancZ[3] (Z basis), flag = flagX[3] (X basis)
// Mirrors your X-stab labeled "Stabilizer 3" with ancX[3]:
// =========================
cx  q[14], ancZ[3];
cx  flagX[3], ancZ[3];
cx  q[10], ancZ[3];
cx  q[12], ancZ[3];
cx  q[6],  ancZ[3];
cx  q[8],  ancZ[3];
cx  q[4],  ancZ[3];
cx  q[2],  ancZ[3];
cx  flagX[3], ancZ[3];
cx  q[0],  ancZ[3];



barrier q, ancX, ancZ, flagX, flagZ;


