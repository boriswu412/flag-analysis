OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancZ[9];
qreg flagX[6];

// g9 g3 g6

cx q[12], ancZ[6];    // q17 m1
cx flagX[5], ancZ[6]; // f2 m1
cx q[7], ancZ[6];     // q11 m1
cx flagX[4], ancZ[6]; // f1 m1
cx q[4], ancZ[6];     // q10 m1

cx q[12], ancZ[7];    // q17 m2
cx q[10], ancZ[8];    // q9 m3

cx flagX[5], ancZ[7]; // f2 m2
cx flagX[4], ancZ[8]; // f1 m3
cx q[6], ancZ[6];     // q8 m1
cx q[13], ancZ[7];    // q18 m2
cx q[11], ancZ[8];    // q16 m3

cx q[14], ancZ[7];    // q19 m2
cx q[14], ancZ[8];    // q19 m3
cx flagX[5], ancZ[7]; // f2 m2
cx q[11], ancZ[7];    // q16 m2
cx flagX[4], ancZ[8]; // f1 m3

cx flagX[5], ancZ[6]; // f2 m1
cx q[10], ancZ[8];    // q6 m3
cx q[11], ancZ[6];    // q9 m1
cx flagX[4], ancZ[6]; // f1 m1
cx q[10], ancZ[6];    // q16 m1