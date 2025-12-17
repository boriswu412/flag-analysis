OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];          // data qubits
qreg ancX[1];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[0];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[0];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[1];      // flags measured in Z (paired with X-type stabs)




// =========================
// Stabilizer 1 : ( YXXYI )
// ancilla = ancX[0] (X basis), flag = flagZ[0] (Z basis)
// =========================
cz  ancX[0], q[1];
cx  ancX[0], flagZ[0];
cy  ancX[0], q[0];
cy  ancX[0], q[3];
cx  ancX[0], flagZ[0];
cx  ancX[0], q[2];

barrier;
