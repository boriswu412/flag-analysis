OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancX[9];
qreg flagZ[6];

cx ancX[0], q[1];      // q2 m1 g10
cx ancX[0], flagZ[1]; // f2 m1  
cx ancX[0], q[8];      // q9 m1 g10
cx ancX[0], flagZ[0]; // f1 m1 
cx ancX[0], q[7];      // q8 m1 g10
cx ancX[1], q[1];      // q3 m2 g11
cx ancX[2], q[4];      // q5 m3 g16

cx ancX[1], flagZ[1]; // f2 m2
cx ancX[2], flagZ[0]; // f1 m3
cx ancX[0], q[5];      // q6 m1 g10
cx ancX[1], q[3];      // q4 m2 g11
cx ancX[2], q[0];      // q1 m3 g16
cx ancX[1], q[2];      // q3 m2 g11
cx ancX[2], q[2];      // q3 m3 g16
cx ancX[1], flagZ[1]; // f2 m2
cx ancX[1], q[0];      // q1 m2 g11
cx ancX[2], flagZ[0]; // f1 m3
cx ancX[0], flagZ[1]; // f2 m1
cx ancX[2], q[6];      // q7 m3 g16
cx ancX[0], q[4];      // q5 m1 g10
cx ancX[0], flagZ[0]; // f1 m1
cx ancX[0], q[0];      // q1 m1 g10