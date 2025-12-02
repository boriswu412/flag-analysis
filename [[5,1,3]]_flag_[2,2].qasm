OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];          
qreg ancX[4];       
qreg ancZ[0];       
qreg flagX[0];      
qreg flagZ[2];     




// BEGIN PERMUTE
cz q[1], ancX[0];
cx ancX[1], q[2];
cx ancX[0], flagZ[0];
cz q[2], ancX[0];
cx ancX[1], flagZ[0];
cx ancX[0], q[3] ;
cz q[4], ancX[1];
cx ancX[1], q[0];
cx ancX[0], flagZ[0];
cx ancX[0], q[0];
cx ancX[1], flagZ[0];
cz q[3], ancX[1];
cz q[0], ancX[2];
cx ancX[3], q[3];
cx ancX[2], flagZ[1];
cz q[1], ancX[2];
cx ancX[3], flagZ[1];
cz q[0], ancX[3];
cx ancX[2], q[2];
cx ancX[3], q[1];
cx ancX[2], flagZ[1];
cx ancX[2], q[4];
cx ancX[3], flagZ[1];
cz q[4], ancX[3];
// END PERMUTE