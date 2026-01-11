OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancX[9];
qreg flagZ[9];

////////////////////////////////////////////////
// Block 1
////////////////////////////////////////////////
cx ancX[3], q[0];   // m4 q1
cx ancX[7], q[7];   // m8 q8
cx ancX[8], q[4];   // m9 q5

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 2
////////////////////////////////////////////////
cx ancX[0], q[0];        // m1 q1
cx ancX[3], flagZ[0];   // m4 f1
cx ancX[4], q[8];       // m5 q9
cx ancX[7], flagZ[3];   // m8 f4
cx ancX[8], flagZ[4];   // m9 f5

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 3
////////////////////////////////////////////////
cx ancX[0], flagZ[0];   // m1 f1
cx ancX[2], q[11];      // m3 q12
cx ancX[3], q[1];       // m4 q2
cx ancX[4], flagZ[6];   // m5 f7
cx ancX[7], q[8];       // m8 q9
cx ancX[8], q[6];       // m9 q7

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 4
////////////////////////////////////////////////
cx ancX[0], q[1];       // m1 q2
cx ancX[2], flagZ[4];   // m3 f5
cx ancX[3], flagZ[1];   // m4 f2
cx ancX[4], q[5];       // m5 q6
cx ancX[5], q[15];      // m6 q16
cx ancX[6], q[9];       // m7 q10
cx ancX[7], flagZ[7];   // m8 f8
cx ancX[8], flagZ[5];   // m9 f6

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 5
////////////////////////////////////////////////
cx ancX[0], q[2];       // m1 q3
cx ancX[1], q[0];       // m2 q1
cx ancX[2], q[12];      // m3 q13
cx ancX[3], q[4];       // m4 q5
cx ancX[4], q[15];      // m5 q16
cx ancX[5], flagZ[6];   // m6 f7
cx ancX[7], q[10];      // m8 q11
cx ancX[8], q[7];       // m9 q8

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 6
////////////////////////////////////////////////
cx ancX[0], flagZ[0];   // m1 f1
cx ancX[1], flagZ[2];   // m2 f3
cx ancX[2], q[13];      // m3 q14
cx ancX[3], q[5];       // m4 q6
cx ancX[4], flagZ[6];   // m5 f7
cx ancX[5], q[18];      // m6 q19
cx ancX[6], flagZ[8];   // m7 f9
cx ancX[7], q[9];       // m8 q10
cx ancX[8], q[10];      // m9 q11

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 7
////////////////////////////////////////////////
cx ancX[0], q[3];       // m1 q4
cx ancX[1], q[2];       // m2 q3
cx ancX[2], flagZ[4];   // m3 f5
cx ancX[3], flagZ[0];   // m4 f1
cx ancX[4], q[18];      // m5 q19
cx ancX[5], q[16];      // m6 q17
cx ancX[6], q[11];      // m7 q12
cx ancX[7], flagZ[3];   // m8 f4
cx ancX[8], flagZ[4];   // m9 f5

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 8
////////////////////////////////////////////////
cx ancX[1], q[4];       // m2 q5
cx ancX[2], q[14];      // m3 q15
cx ancX[3], q[7];       // m4 q8
cx ancX[5], flagZ[6];   // m6 f7
cx ancX[6], q[10];      // m7 q11
cx ancX[7], q[15];      // m8 q16
cx ancX[8], q[11];      // m9 q12

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 9
////////////////////////////////////////////////
cx ancX[1], flagZ[2];   // m2 f3
cx ancX[3], flagZ[1];   // m4 f2
cx ancX[6], flagZ[8];   // m7 f9
cx ancX[7], flagZ[7];   // m8 f8
cx ancX[8], flagZ[5];   // m9 f6

barrier q, ancX, flagZ;

////////////////////////////////////////////////
// Block 10
////////////////////////////////////////////////
cx ancX[1], q[6];       // m2 q7
cx ancX[3], ancX[8];    // m4 m9
cx ancX[5], q[17];      // m6 q18
cx ancX[6], q[14];      // m7 q15
cx ancX[7], q[16];      // m8 q17
cx ancX[8], q[12];      // m9 q13