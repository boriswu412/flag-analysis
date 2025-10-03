OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];          // data qubits
qreg ancX[3];       // ancillas measured in X (for X-type stabilizers)
qreg ancZ[3];       // ancillas measured in Z (for Z-type stabilizers)
qreg flagX[3];      // flags measured in X (paired with Z-type stabs)
qreg flagZ[3];      // flags measured in Z (paired with X-type stabs)

creg syn[6];        // syndrome bits (one per stabilizer)
creg flgX[3];       // flag bits for X-basis flags  (used in stabs 4..6)
creg flgZ[3];       // flag bits for Z-basis flags  (used in stabs 1..3)

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancX[0]  -> syn[0];
measure flagZ[0] -> flgZ[0];

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancX[1]  -> syn[1];
measure flagZ[1] -> flgZ[1];

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancX[2]  -> syn[2];
measure flagZ[2] -> flgZ[2];

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancZ[0]  -> syn[3];
measure flagX[0] -> flgX[0];

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancZ[1]  -> syn[4];
measure flagX[1] -> flgX[1];

barrier q, ancX, ancZ, flagX, flagZ;

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
measure ancZ[2]  -> syn[5];
measure flagX[2] -> flgX[2];

barrier q, ancX, ancZ, flagX, flagZ;