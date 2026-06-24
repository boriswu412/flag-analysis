OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
qreg ancZ[3];

// ========== Z-type stabilizers (raw, sequential) ==========

// --- IIIIZZZZ (ancZ[0]) ---
cx q[3], ancZ[0];
cx q[4], ancZ[0];
cx q[5], ancZ[0];
cx q[6], ancZ[0];
barrier q, ancZ;

// --- IZZIIZZZ (ancZ[1]) ---
cx q[1], ancZ[1];
cx q[2], ancZ[1];
cx q[5], ancZ[1];
cx q[6], ancZ[1];
barrier q, ancZ;

// --- ZIZIZIZI (ancZ[2]) ---
cx q[0], ancZ[2];
cx q[2], ancZ[2];
cx q[4], ancZ[2];
cx q[6], ancZ[2];
barrier q, ancZ;
