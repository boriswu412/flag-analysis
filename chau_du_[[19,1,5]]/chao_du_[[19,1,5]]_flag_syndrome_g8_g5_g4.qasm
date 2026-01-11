OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancZ[9];
qreg flagX[6];

cx q[16], ancZ[3];   // q17 m1
cx flagX[3], ancZ[3]; // f2 m1
cx q[10], ancZ[3];   // q11 m1
cx flagX[2], ancZ[3]; // f1 m1
cx q[9], ancZ[3];    // q10 m1

cx q[16], ancZ[4];   // q17 m2
cx q[8], ancZ[5];    // q9 m3

cx flagX[3], ancZ[4]; // f2 m2
cx flagX[2], ancZ[5]; // f1 m3
cx q[7], ancZ[3];    // q8 m1
cx q[17], ancZ[4];   // q18 m2
cx q[15], ancZ[5];   // q16 m3

cx q[18], ancZ[4];   // q19 m2
cx q[18], ancZ[5];   // q19 m3
cx flagX[3], ancZ[4]; // f2 m2
cx q[15], ancZ[4];   // q16 m2
cx flagX[2], ancZ[5]; // f1 m3

cx flagX[3], ancZ[3]; // f2 m1
cx q[5], ancZ[5];    // q6 m3
cx q[8], ancZ[3];    // q9 m1
cx flagX[2], ancZ[3]; // f1 m1
cx q[15], ancZ[3];   // q16 m1