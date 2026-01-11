OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];      // data 1..19
qreg ancX[9];    // was ancZ
qreg flagZ[12];  // was flagX

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