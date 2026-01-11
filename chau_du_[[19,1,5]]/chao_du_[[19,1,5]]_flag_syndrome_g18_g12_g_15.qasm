OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancX[9];
qreg flagZ[6];

//g18 g12 g15

cx ancX[6], q[12];    // q17 m1
cx ancX[6], flagZ[5]; // f2 m1
cx ancX[6], q[7];     // q11 m1
cx ancX[6], flagZ[4]; // f1 m1
cx ancX[6], q[4];     // q10 m1

cx ancX[7], q[12];    // q17 m2
cx ancX[8], q[10];    // q9 m3

cx ancX[7], flagZ[5]; // f2 m2
cx ancX[8], flagZ[4]; // f1 m3
cx ancX[6], q[6];     // q8 m1
cx ancX[7], q[13];    // q18 m2
cx ancX[8], q[11];    // q16 m3

cx ancX[7], q[14];    // q19 m2
cx ancX[8], q[14];    // q19 m3
cx ancX[7], flagZ[5]; // f2 m2
cx ancX[7], q[11];    // q16 m2
cx ancX[8], flagZ[4]; // f1 m3

cx ancX[6], flagZ[5]; // f2 m1
cx ancX[6], q[10];    // q6 m3
cx ancX[6], q[11];    // q9 m1
cx ancX[6], flagZ[4]; // f1 m1
cx ancX[8], q[9];    // q16 m1

