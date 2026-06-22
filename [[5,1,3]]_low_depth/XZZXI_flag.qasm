OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];          // data qubits
qreg ancX[1];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[0];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[0];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[1];      // flags measured in Z (paired with X-type stabs)





//barrier q, ancX, ancZ, flagX, flagZ;

// =========================
// Stabilizer 1 :  XZZXI
// ancilla = ancZ[0] (Z basis), flag = flagX[0] (X basis)
// =========================
cx  ancX[0], q[0];
cx ancX[0], flagZ[0];
cz  ancX[0], q[1];
cz  ancX[0], q[2];
cx ancX[0], flagZ[0];
cx  ancX[0], q[3];


