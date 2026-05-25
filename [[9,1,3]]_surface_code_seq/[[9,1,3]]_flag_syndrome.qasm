OPENQASM 2.0;
include "qelib1.inc";

qreg q[9];          // data qubits
qreg ancX[4];       // ancillas measured in X (X-type checks)
qreg ancZ[4];       // ancillas measured in Z (Z-type checks)
qreg flagX[2];      // flags measured in X (for 4-body Z checks)
qreg flagZ[2];      // flags measured in Z (for 4-body X checks)

// [[9,1,3]] stabilizers from [[9,1,3]].txt
// X checks:
//  1) q0 q1
//  2) q1 q2 q4 q5   (flagged)
//  3) q3 q4 q6 q7   (flagged)
//  4) q7 q8
// Z checks:
//  1) q0 q1 q3 q4   (flagged)
//  2) q2 q5
//  3) q3 q6
//  4) q4 q5 q7 q8   (flagged)

// X-type stabilizers
// X1: X on q0,q1
cx ancX[0], q[0];
cx ancX[0], q[1];

// X2: X on q1,q2,q4,q5 (with flagZ[0])
cx ancX[1], q[1];
cx ancX[1], flagZ[0];
cx ancX[1], q[2];
cx ancX[1], q[4];
cx ancX[1], flagZ[0];
cx ancX[1], q[5];

// X3: X on q3,q4,q6,q7 (with flagZ[1])
cx ancX[2], q[3];
cx ancX[2], flagZ[1];
cx ancX[2], q[4];
cx ancX[2], q[6];
cx ancX[2], flagZ[1];
cx ancX[2], q[7];

// X4: X on q7,q8
cx ancX[3], q[7];
cx ancX[3], q[8];

// Z-type stabilizers
// Z1: Z on q0,q1,q3,q4 (with flagX[0])
cx q[0], ancZ[0];
cx flagX[0], ancZ[0];
cx q[1], ancZ[0];
cx q[3], ancZ[0];
cx flagX[0], ancZ[0];
cx q[4], ancZ[0];

// Z2: Z on q2,q5
cx q[2], ancZ[1];
cx q[5], ancZ[1];

// Z3: Z on q3,q6
cx q[3], ancZ[2];
cx q[6], ancZ[2];

// Z4: Z on q4,q5,q7,q8 (with flagX[1])
cx q[4], ancZ[3];
cx flagX[1], ancZ[3];
cx q[5], ancZ[3];
cx q[7], ancZ[3];
cx flagX[1], ancZ[3];
cx q[8], ancZ[3];