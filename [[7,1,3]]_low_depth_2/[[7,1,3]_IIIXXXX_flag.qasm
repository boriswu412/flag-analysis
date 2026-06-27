OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];          // data qubits
qreg ancX[1];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[0];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[0];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[1];      // flags measured in Z (paired with X-type stabs)




// =========================
// Stabilizer 1 : (IIIXXXX)
// ancilla = ancX[0] (X basis), flag = flagZ[0] (Z basis)
// =========================
cx ancX[0] ,q[3];
cx ancX[0] ,flagZ[0];
cx ancX[0] ,q[4];
cx ancX[0] ,q[5];
cx ancX[0], flagZ[0];
cx ancX[0], q[6];

barrier;
