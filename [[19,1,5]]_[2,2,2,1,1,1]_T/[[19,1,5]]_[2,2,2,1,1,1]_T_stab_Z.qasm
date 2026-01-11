OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancZ[9];
qreg flagX[9];

////////////////////////////////////////////////
// Block 1
////////////////////////////////////////////////
cx q[0], ancZ[3];   // q1 m4
cx q[7], ancZ[7];   // q8 m8
cx q[4], ancZ[8];   // q5 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 2
////////////////////////////////////////////////
cx q[0], ancZ[0];       // q1 m1
cx flagX[0], ancZ[3];  // f1 m4
cx q[8], ancZ[4];      // q9 m5
cx flagX[3], ancZ[7];  // f4 m8
cx flagX[4], ancZ[8];  // f5 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 3
////////////////////////////////////////////////
cx flagX[0], ancZ[0];  // f1 m1
cx q[11], ancZ[2];     // q12 m3
cx q[1], ancZ[3];      // q2 m4
cx flagX[6], ancZ[4];  // f7 m5
cx q[8], ancZ[7];      // q9 m8
cx q[6], ancZ[8];      // q7 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 4
////////////////////////////////////////////////
cx q[1], ancZ[0];      // q2 m1
cx flagX[4], ancZ[2];  // f5 m3
cx flagX[1], ancZ[3];  // f2 m4
cx q[5], ancZ[4];      // q6 m5
cx q[15], ancZ[5];     // q16 m6
cx q[9], ancZ[6];      // q10 m7
cx flagX[7], ancZ[7];  // f8 m8
cx flagX[5], ancZ[8];  // f6 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 5
////////////////////////////////////////////////
cx q[2], ancZ[0];      // q3 m1
cx q[0], ancZ[1];      // q1 m2
cx q[12], ancZ[2];     // q13 m3
cx q[4], ancZ[3];      // q5 m4
cx q[15], ancZ[4];     // q16 m5
cx flagX[6], ancZ[5];  // f7 m6
cx q[10], ancZ[7];     // q11 m8
cx q[7], ancZ[8];      // q8 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 6
////////////////////////////////////////////////
cx flagX[0], ancZ[0];  // f1 m1
cx flagX[2], ancZ[1];  // f3 m2
cx q[13], ancZ[2];     // q14 m3
cx q[5], ancZ[3];      // q6 m4
cx flagX[6], ancZ[4];  // f7 m5
cx q[18], ancZ[5];     // q19 m6
cx flagX[8], ancZ[6];  // f9 m7
cx q[9] , ancZ[7];       // q10 m8
cx q[10], ancZ[8];     // q11 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 7
////////////////////////////////////////////////
cx q[3], ancZ[0];      // q4 m1
cx q[2], ancZ[1];      // q3 m2
cx flagX[4], ancZ[2];  // f5 m3
cx flagX[0], ancZ[3];  // f1 m4
cx q[18], ancZ[4];     // q19 m5
cx q[16], ancZ[5];     // q17 m6
cx q[11], ancZ[6];     // q12 m7
cx flagX[3], ancZ[7];  // f4 m8
cx flagX[4], ancZ[8];  // f5 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 8
////////////////////////////////////////////////
cx q[4], ancZ[1];      // q5 m2
cx q[14], ancZ[2];     // q15 m3
cx q[7], ancZ[3];      // q8 m4
cx flagX[6], ancZ[5];  // f7 m6
cx q[10], ancZ[6];     // q11 m7
cx q[15], ancZ[7];     // q16 m8
cx q[11], ancZ[8];     // q12 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 9
////////////////////////////////////////////////
cx flagX[2], ancZ[1];  // f3 m2
cx flagX[1], ancZ[3];  // f2 m4
cx flagX[8], ancZ[6];  // f9 m7
cx flagX[7], ancZ[7];  // f8 m8
cx flagX[5], ancZ[8];  // f6 m9

barrier q, ancZ, flagX;

////////////////////////////////////////////////
// Block 10
////////////////////////////////////////////////
cx q[6], ancZ[1];      // q7 m2
cx q[8], ancZ[3];     // q9 m4
cx q[17], ancZ[5];     // q18 m6
cx q[14], ancZ[6];     // q15 m7
cx q[16], ancZ[7];     // q17 m8
cx q[12], ancZ[8];     // q13 m9