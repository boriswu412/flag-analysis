OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
qreg ancZ[9];
qreg flagX[6];
qreg ancX[9];
qreg flagZ[6];

//g7 g1 g2

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

//---------------------------------------------

//g8 g5 g4

cx q[16], ancZ[3];   // q17 m1 g8
cx flagX[3], ancZ[3]; // f2 m1
cx q[10], ancZ[3];   // q11 m1 g8
cx flagX[2], ancZ[3]; // f1 m1
cx q[9], ancZ[3];    // q10 m1 g8
cx q[16], ancZ[4];   // q17 m2
cx q[8], ancZ[5];    // q9 m3
cx flagX[3], ancZ[4]; // f2 m2
cx flagX[2], ancZ[5]; // f1 m3
cx q[7], ancZ[3];    // q8 m1 g8
cx q[17], ancZ[4];   // q18 m2
cx q[15], ancZ[5];   // q16 m3
cx q[18], ancZ[4];   // q19 m2
cx q[18], ancZ[5];   // q19 m3
cx flagX[3], ancZ[4]; // f2 m2
cx q[15], ancZ[4];   // q16 m2
cx flagX[2], ancZ[5]; // f1 m3
cx flagX[3], ancZ[3]; // f2 m1
cx q[5], ancZ[5];    // q6 m3
cx q[8], ancZ[3];    // q9 m1 g8
cx flagX[2], ancZ[3]; // f1 m1
cx q[15], ancZ[3];   // q16 m1 g8

//---------------------------------------------
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
cx q[10], ancZ[6];    // q6 m3
cx q[11], ancZ[6];    // q9 m1
cx flagX[4], ancZ[6]; // f1 m1
cx q[9], ancZ[8];    // q16 m1
//---------------------------------------------
//g16 g10 g11

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

//---------------------------------------------
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


//---------------------------------------------
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

