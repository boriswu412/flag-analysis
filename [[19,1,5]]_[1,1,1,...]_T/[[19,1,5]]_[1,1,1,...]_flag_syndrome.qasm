OPENQASM 2.0;
include "qelib1.inc";

// --- Register Declarations ---
qreg q[19];      // Data qubits
qreg ancZ[9];    // Z-type ancillas (m1-m9)
qreg ancX[9];    // X-type ancillas
qreg flagX[12];  // X-type flags (f1-f12)
qreg flagZ[12];  // Z-type flags

// ---- control/target swapped (back) ----
cx q[0], ancZ[0];
cx q[2], ancZ[1];
cx q[11], ancZ[2];
cx q[5], ancZ[4];
cx q[15], ancZ[5];
cx q[9], ancZ[6];
cx q[1], ancZ[3];
cx q[8], ancZ[7];
cx q[4], ancZ[8];

barrier;

cx flagX[0], ancZ[0];
cx flagX[1], ancZ[1];
cx flagX[2], ancZ[2];
cx flagX[3], ancZ[4];
cx flagX[4], ancZ[5];
cx flagX[5], ancZ[6];
cx flagX[6], ancZ[3];
cx flagX[8], ancZ[7];
cx flagX[10], ancZ[8];

barrier;

cx q[1], ancZ[0];
cx q[0], ancZ[1];
cx q[12], ancZ[2];
cx q[8], ancZ[4];
cx q[16], ancZ[5];
cx q[10], ancZ[6];
cx q[4], ancZ[3];
cx q[7], ancZ[7];
cx q[6], ancZ[8];

barrier;

cx q[2], ancZ[0];
cx q[4], ancZ[1];
cx q[13], ancZ[2];
cx q[15], ancZ[4];
cx q[18], ancZ[5];
cx q[14], ancZ[6];
cx flagX[7], ancZ[3];
cx flagX[9], ancZ[7];
cx flagX[11], ancZ[8];

barrier;

cx flagX[0], ancZ[0];
cx flagX[1], ancZ[1];
cx flagX[2], ancZ[2];

cx flagX[3], ancZ[4];
cx flagX[4], ancZ[5];
cx flagX[5], ancZ[6];

cx q[0], ancZ[3];
cx q[10], ancZ[7];
cx q[7], ancZ[8];


barrier;


cx q[3], ancZ[0];
cx q[6], ancZ[1];
cx q[14], ancZ[2];
cx q[18], ancZ[4];
cx q[17], ancZ[5];
cx q[11], ancZ[6];
cx q[5], ancZ[3];
cx q[9], ancZ[7];
cx q[10], ancZ[8];

barrier;

cx flagX[6], ancZ[3];
cx flagX[8], ancZ[7];
cx flagX[10], ancZ[8];

barrier;

cx q[7], ancZ[3];
cx q[15], ancZ[7];
cx q[11], ancZ[8];

barrier;

cx flagX[7], ancZ[3];
cx flagX[9], ancZ[7];
cx flagX[11], ancZ[8];

barrier;


cx q[8], ancZ[3];
cx q[16], ancZ[7];
cx q[12], ancZ[8];

barrier;

// ---- control/target swapped ----
cx ancX[0], q[0];
cx ancX[1], q[2];
cx ancX[2], q[11];
cx ancX[4], q[5];
cx ancX[5], q[15];
cx ancX[6], q[9];
cx ancX[3], q[1];
cx ancX[7], q[8];
cx ancX[8], q[4];

barrier;

cx ancX[0], flagZ[0];
cx ancX[1], flagZ[1];
cx ancX[2], flagZ[2];
cx ancX[4], flagZ[3];
cx ancX[5], flagZ[4];
cx ancX[6], flagZ[5];
cx ancX[3], flagZ[6];
cx ancX[7], flagZ[8];
cx ancX[8], flagZ[10];

barrier;

cx ancX[0], q[1];
cx ancX[1], q[0];
cx ancX[2], q[12];
cx ancX[4], q[8];
cx ancX[5], q[16];
cx ancX[6], q[10];
cx ancX[3], q[4];
cx ancX[7], q[7];
cx ancX[8], q[6];

barrier;

cx ancX[0], q[2];
cx ancX[1], q[4];
cx ancX[2], q[13];
cx ancX[4], q[15];
cx ancX[5], q[18];
cx ancX[6], q[14];
cx ancX[3], flagZ[7];
cx ancX[7], flagZ[9];
cx ancX[8], flagZ[11];

barrier;

cx ancX[0], flagZ[0];
cx ancX[1], flagZ[1];
cx ancX[2], flagZ[2];

cx ancX[4], flagZ[3];
cx ancX[5], flagZ[4];
cx ancX[6], flagZ[5];

cx ancX[3], q[0];
cx ancX[7], q[10];
cx ancX[8], q[7];

barrier;

cx ancX[0], q[3];
cx ancX[1], q[6];
cx ancX[2], q[14];
cx ancX[4], q[18];
cx ancX[5], q[17];
cx ancX[6], q[11];
cx ancX[3], q[5];
cx ancX[7], q[9];
cx ancX[8], q[10];

barrier;

cx ancX[3], flagZ[6];
cx ancX[7], flagZ[8];
cx ancX[8], flagZ[10];

barrier;

cx ancX[3], q[7];
cx ancX[7], q[15];
cx ancX[8], q[11];

barrier;

cx ancX[3], flagZ[7];
cx ancX[7], flagZ[9];
cx ancX[8], flagZ[11];

barrier;

cx ancX[3], q[8];
cx ancX[7], q[16];
cx ancX[8], q[12];

barrier;