OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];          // data qubits
qreg ancX[1];       // ancillas measured in X (for X-type stabilizers)




// =========================
// Stabilizer 1 : ( YXXYI )
// ancilla = ancX[0] (X basis)
// =========================
cx  ancX[0], q[1];
cy  ancX[0], q[0];
cy  ancX[0], q[3];
cx  ancX[0], q[2];

barrier;
