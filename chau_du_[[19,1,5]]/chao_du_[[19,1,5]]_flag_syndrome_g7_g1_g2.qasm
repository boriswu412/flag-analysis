OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancZ[9];
qreg flagX[6];

cx q[1], ancZ[0];      // q2 m1 g1
cx flagX[1], ancZ[0]; // f2 m1  
cx q[8], ancZ[0];      // q9 m1 g1
cx flagX[0], ancZ[0]; // f1 m1 
cx q[7], ancZ[0];      // q8 m1 g1
cx  q[1], ancZ[1];      // q3 m2 g2
cx q[4] , ancZ[2];      // q5 m3 g7

cx flagX[1], ancZ[1]; // f2 m2
cx flagX[0], ancZ[2]; // f1 m3
cx q[5], ancZ[0];      // q6 m1 g1
cx q[3], ancZ[1];      // q4 m2 g2
cx q[0], ancZ[2];      // q1 m3 g7
cx q[2], ancZ[1];      // q3 m2 g2
cx q[2], ancZ[2];      // q3 m3 g7
cx flagX[1], ancZ[1]; // f2 m2
cx q[0], ancZ[1];      // q1 m2 g2
cx flagX[0], ancZ[2]; // f1 m3
cx flagX[1], ancZ[0]; // f2 m1
cx q[6], ancZ[2];      // q7 m3 g7
cx q[4], ancZ[0];      // q5 m1 g1
cx flagX[0], ancZ[0]; // f1 m1
cx q[0], ancZ[0];      // q1 m1 g1