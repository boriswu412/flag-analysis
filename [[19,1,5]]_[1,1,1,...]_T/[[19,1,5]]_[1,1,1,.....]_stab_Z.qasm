OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];      // data 1..19  -> q[0]..q[18]
qreg ancZ[9];    // m1..m9     -> ancZ[0]..ancZ[8]
qreg flagX[12];  // f1..f12    -> flagX[0]..flagX[11]

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