OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancX[9];
qreg flagZ[6];

//g17 g14 g13

cx ancX[3], q[16];    // q17 m1
cx ancX[3], flagZ[3]; // f2 m1
cx ancX[3], q[10];    // q11 m1
cx ancX[3], flagZ[2]; // f1 m1
cx ancX[3], q[9];     // q10 m1

cx ancX[4], q[16];    // q17 m2
cx ancX[5], q[8];     // q9 m3

cx ancX[4], flagZ[3]; // f2 m2
cx ancX[5], flagZ[2]; // f1 m3
cx ancX[3], q[7];     // q8 m1
cx ancX[4], q[17];    // q18 m2
cx ancX[5], q[15];    // q16 m3

cx ancX[4], q[18];    // q19 m2
cx ancX[5], q[18];    // q19 m3
cx ancX[4], flagZ[3]; // f2 m2
cx ancX[4], q[15];    // q16 m2
cx ancX[5], flagZ[2]; // f1 m3

cx ancX[3], flagZ[3]; // f2 m1
cx ancX[5], q[5];     // q6 m3
cx ancX[3], q[8];     // q9 m1
cx ancX[3], flagZ[2]; // f1 m1
cx ancX[3], q[15];    // q16 m1