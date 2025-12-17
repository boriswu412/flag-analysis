OPENQASM 2.0;
include "qelib1.inc";

// --- Register Declarations ---
qreg q[19];      // Data qubits
qreg ancZ[9];    // Z-type ancillas (m1-m9)
qreg ancX[9];    // X-type ancillas
qreg flagX[12];  // X-type flags (f1-f12)
qreg flagZ[12];  // Z-type flags

// =========================================================
// SECTION 1: Z-TYPE STABILIZERS (Data controls Ancilla)
// =========================================================

// --- BLOCK 1 ---
cx q[0], ancZ[0]; cx q[2], ancZ[1]; cx q[11], ancZ[2];
cx q[1], ancZ[3]; cx q[5], ancZ[4]; cx q[15], ancZ[5];
cx q[9], ancZ[6]; cx q[8], ancZ[7]; cx q[4], ancZ[8];

// --- BLOCK 2 ---
cx ancZ[0], flagX[0]; cx ancZ[1], flagX[1]; cx ancZ[2], flagX[2];
cx ancZ[4], flagX[3]; cx ancZ[5], flagX[4]; cx ancZ[6], flagX[5];
cx ancZ[7], flagX[6]; cx ancZ[8], flagX[8];

cx q[1], ancZ[0]; cx q[0], ancZ[1]; cx q[12], ancZ[2];
cx q[3], ancZ[3]; cx q[16], ancZ[4]; cx q[6], ancZ[5];
cx q[10], ancZ[6]; cx q[7], ancZ[7]; cx q[5], ancZ[8];

// --- BLOCK 3 ---
cx ancZ[0], flagX[0]; cx ancZ[1], flagX[1]; cx ancZ[2], flagX[2];
cx ancZ[3], flagX[7]; cx ancZ[4], flagX[3]; cx ancZ[5], flagX[9];
cx ancZ[6], flagX[10]; cx ancZ[7], flagX[4]; cx ancZ[8], flagX[11];

cx q[2], ancZ[0]; cx q[3], ancZ[2]; cx q[8], ancZ[3];
cx q[13], ancZ[4]; cx q[18], ancZ[6]; cx q[4], ancZ[5];

// --- BLOCK 4 ---
cx ancZ[0], flagX[0]; cx ancZ[1], flagX[1]; cx ancZ[2], flagX[2];
cx ancZ[3], flagX[3]; cx ancZ[4], flagX[4]; cx ancZ[5], flagX[5];
cx ancZ[6], flagX[6]; cx ancZ[7], flagX[7]; cx ancZ[8], flagX[8];

barrier q, ancZ, flagX, ancX, flagZ;

// =========================================================
// SECTION 2: X-TYPE STABILIZERS (Roles Swapped)
// =========================================================

// --- BLOCK 1: Ancilla controls Data ---
cx ancX[0], q[0]; cx ancX[1], q[2]; cx ancX[2], q[11];
cx ancX[3], q[1]; cx ancX[4], q[5]; cx ancX[5], q[15];
cx ancX[6], q[9]; cx ancX[7], q[8]; cx ancX[8], q[4];

// --- BLOCK 2: Flag controls Ancilla ---
cx flagZ[0], ancX[0]; cx flagZ[1], ancX[1]; cx flagZ[2], ancX[2];
cx flagZ[3], ancX[4]; cx flagZ[4], ancX[5]; cx flagZ[5], ancX[6];
cx flagZ[6], ancX[7]; cx flagZ[8], ancX[8];

cx ancX[0], q[1]; cx ancX[1], q[0]; cx ancX[2], q[12];
cx ancX[3], q[3]; cx ancX[4], q[16]; cx ancX[5], q[6];
cx ancX[6], q[10]; cx ancX[7], q[7]; cx ancX[8], q[5];

// --- BLOCK 3: Flag controls Ancilla ---
cx flagZ[0], ancX[0]; cx flagZ[1], ancX[1]; cx flagZ[2], ancX[2];
cx flagZ[7], ancX[3]; cx flagZ[3], ancX[4]; cx flagZ[9], ancX[5];
cx flagZ[10], ancX[6]; cx flagZ[4], ancX[7]; cx flagZ[11], ancX[8];

cx ancX[0], q[2]; cx ancX[2], q[3]; cx ancX[3], q[8];
cx ancX[4], q[13]; cx ancX[6], q[18]; cx ancX[5], q[4];

// --- BLOCK 4: Final Flag Connections ---
cx flagZ[0], ancX[0]; cx flagZ[1], ancX[1]; cx flagZ[2], ancX[2];
cx flagZ[3], ancX[3]; cx flagZ[4], ancX[5]; cx flagZ[5], ancX[6];