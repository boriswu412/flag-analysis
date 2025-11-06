OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];          // data qubits
qreg ancX[3];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[3];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[3];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[3];      // flags measured in Z (paired with X-type stabs)



// =========================
// Stabilizer 1 : ( I I I X X X X )
// ancilla = ancX[0] (X basis), flag = flagZ[0] (Z basis)
// =========================
cx  ancX[0], q[3];
cx  ancX[0], flagZ[0];
cx  ancX[0], q[4];
cx  ancX[0], q[5];
cx  ancX[0], flagZ[0];
cx  ancX[0], q[6];

barrier;

// =========================
// Stabilizer 2 : ( I X X I I X X )
// ancilla = ancX[1] (X basis), flag = flagZ[1] (Z basis)
// =========================
cx  ancX[1], q[1];
cx  ancX[1], flagZ[1];
cx  ancX[1], q[2];
cx  ancX[1], q[5];
cx  ancX[1], flagZ[1];
cx  ancX[1], q[6];

barrier;

// =========================
// Stabilizer 3 : ( X I X I X I X )
// ancilla = ancX[2] (X basis), flag = flagZ[2] (Z basis)
// =========================
cx  ancX[2], q[0];
cx  ancX[2], flagZ[2];
cx  ancX[2], q[2];
cx  ancX[2], q[4];
cx  ancX[2], flagZ[2];
cx  ancX[2], q[6];

barrier;

// =========================
// Stabilizer 4 : ( I I I Z Z Z Z )
// ancilla = ancZ[0] (Z basis), flag = flagX[0] (X basis)
// =========================
cx  q[3],     ancZ[0];
cx  flagX[0], ancZ[0];
cx  q[4],     ancZ[0];
cx  q[5],     ancZ[0];
cx  flagX[0], ancZ[0];
cx  q[6],     ancZ[0];

barrier;

// =========================
// Stabilizer 5 : ( I Z Z I I Z Z )
// ancilla = ancZ[1] (Z basis), flag = flagX[1] (X basis)
// =========================
cx  q[1],     ancZ[1];
cx  flagX[1], ancZ[1];
cx  q[2],     ancZ[1];
cx  q[5],     ancZ[1];
cx  flagX[1], ancZ[1];
cx  q[6],     ancZ[1];

barrier;


// =========================
// Stabilizer 6 : ( Z I Z I Z I Z )
// ancilla = ancZ[2] (Z basis), flag = flagX[2] (X basis)
// =========================
cx  q[0],     ancZ[2];
cx  flagX[2], ancZ[2];
cx  q[2],     ancZ[2];
cx  q[4],     ancZ[2];
cx  flagX[2], ancZ[2];
cx  q[6],     ancZ[2];

barrier;