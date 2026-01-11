OPENQASM 2.0;
include "qelib1.inc";

// --- Register Declarations ---
qreg q[19];      // Data qubits
qreg ancZ[9];    // Z-type ancillas (m1-m9)
qreg ancX[9];    // X-type ancillas
qreg flagX[9];  // X-type flags (f1-f12)
qreg flagZ[9];  // Z-type flags


//stab_Z
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


//stab_X
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
cx ancX[3], q[8];    // m4 q9
cx ancX[5], q[17];      // m6 q18
cx ancX[6], q[14];      // m7 q15
cx ancX[7], q[16];      // m8 q17
cx ancX[8], q[12];      // m9 q13