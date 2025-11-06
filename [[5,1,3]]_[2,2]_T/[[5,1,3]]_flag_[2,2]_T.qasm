OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];          // data qubits
qreg ancX[4];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[0];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[0];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[2];      // flags measured in Z (paired with X-type stabs)






cz q[0], ancX[2];
cz q[1], ancX[0];
cx ancX[1], q[2];
cx ancX[3], q[3];

barrier q, ancX, ancZ, flagX, flagZ;

cx ancX[0], flagZ[0];
cx ancX[2], flagZ[1];

barrier q, ancX, ancZ, flagX, flagZ;

cz q[1], ancX[2]; 
cx ancX[3], flagZ[1];
cz q[2], ancX[0]; 
cx ancX[1], flagZ[0];

barrier q, ancX, ancZ, flagX, flagZ;

cz q[0], ancX[3]; 
cx ancX[2], q[2];
cx ancX[0], q[3];
cz  q[4], ancX[1];

barrier q, ancX, ancZ, flagX, flagZ;

cx ancX[1], q[0];
cx ancX[3], q[1];
cx ancX[0], flagZ[0];
cx ancX[2], flagZ[1];

barrier q, ancX, ancZ, flagX, flagZ;

cx ancX[0], q[0];
cx ancX[1], flagZ[0];
cx ancX[3], flagZ[1];
cx ancX[2], q[4];

barrier q, ancX, ancZ, flagX, flagZ;

cz q[3], ancX[1];
cz q[4], ancX[3];

barrier q, ancX, ancZ, flagX, flagZ;