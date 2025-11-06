# SAT-based flag qubit circuit verification tool 
## Installation 
Copy this to command line 
```bash
git clone https://github.com/boriswu412/flag-analysis.git
```
Copy this to install the requirement 

```bash
pip install -r requirements.txt
```

## Usage 
In the `config.txt`  there are to rows `qasm_path` and `stab_txt_path`. 

### Qasm file 
#### Qubits
There are five kind of qregs `q` for data qubits, `ancX`, `ancZ` for ancillas which are prepared in X and Z basis, `flagX` ,`flagZ` for flag qubits prepared in X and Z basis. 
