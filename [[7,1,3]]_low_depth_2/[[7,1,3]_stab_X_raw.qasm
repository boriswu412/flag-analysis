OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
qreg ancX[3];

// ========== X-type stabilizers (raw, sequential) ==========

// --- IIIXXXX (ancX[0]) ---
cx ancX[0], q[3];
cx ancX[0], q[4];
cx ancX[0], q[5];
cx ancX[0], q[6];
barrier q, ancX;

// --- IXXIIXX (ancX[1]) ---
cx ancX[1], q[1];
cx ancX[1], q[2];
cx ancX[1], q[5];
cx ancX[1], q[6];
barrier q, ancX;

// --- XIXIXIX (ancX[2]) ---
cx ancX[2], q[0];
cx ancX[2], q[2];
cx ancX[2], q[4];
cx ancX[2], q[6];
barrier q, ancX;
