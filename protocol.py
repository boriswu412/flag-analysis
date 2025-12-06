import json
import os
from typing import Optional, List, Dict

class Condition:
    def __init__(self, cond_type: str, left=None, right=None,
                 operand=None, operands=None):
        self.type = cond_type
        self.left = left
        self.right = right
        self.operand = operand        # for unary
        self.operands = operands      # for n-ary

    def to_dict(self):
        d = {"type": self.type}

        # Leaf binary conditions stay the same
        if self.type in ("equal", "not_equal",
                         "greater", "less", "greater_equal", "less_equal"):
            d["left"] = self.left
            d["right"] = self.right
            return d

        # Unary NOT → convert to operands list
        if self.type == "not":
            d["operands"] = [self.operand.to_dict()]
            return d

        # AND / OR → stays operands
        if self.type in ("and", "or"):
            d["operands"] = [op.to_dict() for op in self.operands]
            return d

        raise ValueError(f"Unknown condition type: {self.type}")


class Branch:
    def __init__(self, target: str, condition: Optional[Condition] = None):
        self.target = target
        self.condition = condition

    def to_dict(self):
        return {
            "condition": self.condition.to_dict() if self.condition else None,
            "target": self.target
        }


class Node:
    def __init__(self, node_id: str, instructions: List[str], branches: List[Branch]):
        self.node_id = node_id
        self.instructions = instructions
        self.branches = branches

    def to_dict(self):
        return {
            "instructions": self.instructions,
            "branches": [b.to_dict() for b in self.branches]
        }


class Protocol:
    def __init__(self, start_node: str):
        self.start_node = start_node
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def to_dict(self):
        return {
            "start_node": self.start_node,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }

    def save_to_file(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Protocol saved to {filepath}")



import json
from typing import Dict, List, Optional

# assumes your existing classes:
# class Condition, class Branch, class Node

def condition_from_dict(d: Optional[dict]) -> Optional[Condition]:
    """Rebuild a Condition (possibly nested) from a JSON dict."""
    if d is None:
        return None

    ctype = d["type"]

    # Leaf comparisons
    if ctype in ("equal", "not_equal", "greater", "less", "greater_equal", "less_equal"):
        return Condition(
            cond_type=ctype,
            left=d.get("left"),
            right=d.get("right")
        )

    # NOT: in your JSON, it's encoded as { "type": "not", "operands": [ subcond ] }
    if ctype == "not":
        ops = d.get("operands", [])
        if len(ops) != 1:
            raise ValueError("NOT condition expects exactly one operand")
        sub = condition_from_dict(ops[0])
        return Condition(cond_type="not", operand=sub)

    # AND / OR: { "type": "and", "operands": [ ... ] }
    if ctype in ("and", "or"):
        ops = d.get("operands", [])
        subconds = [condition_from_dict(x) for x in ops]
        return Condition(cond_type=ctype, operands=subconds)

    raise ValueError(f"Unknown condition type in JSON: {ctype}")

# ---- parse Branch from dict ----

def branch_from_dict(d: dict) -> Branch:
    cond_dict = d.get("condition")
    cond = condition_from_dict(cond_dict) if cond_dict is not None else None
    return Branch(
        target=d["target"],
        condition=cond
    )

def node_from_dict(node_id: str, d: dict) -> Node:
    instrs = d.get("instructions", [])
    branches_dicts = d.get("branches", [])
    branches = [branch_from_dict(bd) for bd in branches_dicts]
    return Node(node_id=node_id, instructions=instrs, branches=branches)

from typing import Tuple

def read_path_steps(nodes: dict, path: list):
    steps = []
    for i, nid in enumerate(path):
        if nid not in nodes:
            raise KeyError(f"Node '{nid}' not found in protocol")

        node = nodes[nid]
        cond_to_next = None

        if i + 1 < len(path):
            next_id = path[i+1]
            # find branch leading to next_id
            for br in node.branches:
                if br.target == next_id:
                    cond_to_next = br.condition.to_dict() if br.condition else None
                    break

        steps.append((nid, node.instructions, cond_to_next))
    return steps
# ---- top-level loader ----

def load_protocol(path: str):
    """
    Load a protocol JSON of the form:
    {
        "start_node": "<id>",
        "nodes": {
            "<id>": { "instructions": [...], "branches": [...] },
            ...
        }
    }

    Returns:
      start_node: str
      protocol: Dict[str, Node]
    """
    with open(path, "r") as f:
        raw = json.load(f)

    start_node = raw["start_node"]
    nodes_dict = raw["nodes"]   # dict: node_id -> dict

    protocol: Dict[str, Node] = {}
    for node_id, node_body in nodes_dict.items():
        protocol[node_id] = node_from_dict(node_id, node_body)

    return start_node, protocol
def all_paths_with_conditions_and_instructions(protocol):
    """
    Yields:
      (branch_steps, condition_list, instruction_list)

    where:
      branch_steps      = [(from_node_id, Branch, to_node_id), ...]
      condition_list    = [Condition, Condition, ...]    # one per branch that has a condition
      instruction_list  = [str, str, ...]                # instructions along the path

    - If a branch has no condition, it does NOT add to condition_list.
    - If a path has no conditions at all, condition_list will be [].
    """

    def dfs(node_id, branch_steps, cond_list, instr_list):
        node = protocol.nodes[node_id]

        # Add this node's instructions
        new_instr_list = instr_list + node.instructions

        # Leaf node → complete path
        if not node.branches:
            yield branch_steps, cond_list, new_instr_list
            return

        for br in node.branches:
            # Record this transition: from node_id to br.target
            step = (node_id, br, br.target)

            # Add this branch's condition (if any)
            if br.condition is None:
                new_cond_list = cond_list.copy()
            else:
                new_cond_list = cond_list + [br.condition]

            if br.target is None:
                # Terminal branch (no next state)
                yield branch_steps + [step], new_cond_list, new_instr_list
            else:
                # Continue DFS from the target state
                yield from dfs(
                    br.target,
                    branch_steps + [step],
                    new_cond_list,
                    new_instr_list
                )

    yield from dfs(protocol.start_node, [], [], [])

def build_protocol_d_3_lai() -> Protocol:
    protocol = Protocol(start_node="root")

    condition_s_1_all_zero = Condition(cond_type="equal", left="s_1", right=0)
    condition_f_1_all_zero = Condition(cond_type="equal", left="f_1", right=0)
    condition_1 = Condition("and", operands=[condition_s_1_all_zero, condition_f_1_all_zero])
    # Root node with unconditional branch to flag_measure
    root_node = Node(
        node_id="root",
        instructions=["flag_syndrome"],
        branches=[Branch(target="f_1_s_1_all_zero", condition= condition_1),
                  Branch(target="not_f_1_s_1_all_zero", condition = Condition("not", operand=condition_1))]
    )
    protocol.add_node(root_node)

    f_1_s_1_all_zero = Node(
        node_id="f_1_s_1_all_zero",
        instructions=[], branches=[]
    )
    protocol.add_node(f_1_s_1_all_zero)

    #node f_1_s_1_all_zero
    not_f_1_s_1_all_zero = Node(
        node_id="not_f_1_s_1_all_zero",
        instructions=["raw_syndrome"],
        branches=[Branch(target="ter_2")]
    )
    protocol.add_node(not_f_1_s_1_all_zero)


    # Flag measurement node
    flag_measure_node = Node(
        node_id="ter_2",
        instructions=[],
        branches=[]
    )
    protocol.add_node(flag_measure_node)

    protocol.save_to_file("./protocols/d_3_lai_protocol.json")

    return protocol
def build_protocol_d_5_lai() -> Protocol:
    protocol = Protocol(start_node="root")



    #condition s_1 == 0 
    condition_s_1_all_zero = Condition(
        cond_type="equal",
        left="s_1",
        right=0
    )
    
    #condition flag are all zero
    condition_f_1_all_zero = Condition(cond_type="equal", left="f_1", right=0)
    
    condition_1 = Condition("and", operands=[condition_s_1_all_zero, condition_f_1_all_zero])
    

    # Root node with unconditional branch to flag_measure
    root_node = Node(
        node_id="root",
        instructions=["flag_syndrome"],
        branches=[Branch(target="all_zero", condition= condition_1),
                  Branch(target="not_all_zero", condition = Condition("not", operand=condition_1))]
    )

    protocol.add_node(root_node)


    #node all zero in first round
    round_1_all_zero = Node("all_zero",[], [])

    protocol.add_node(round_1_all_zero)
    

    condition_2 = Condition("equal", left="f_2", right=0)
    #node not all zero in first round
    round_1_not_all_zero = Node(node_id="not_all_zero" ,instructions=["flagged_syndrome"],branches=[Branch(target="flag_2_all_zero", condition = condition_2),
                  Branch(target = "flag_2_not_all_zero", condition = Condition("not", operand=condition_2))])
    
    protocol.add_node(round_1_not_all_zero)

    flag_2_not_all_zero = Node("flag_2_not_all_zero", ["raw_syndrome"], [Branch(target = "ter_1" )])
    protocol.add_node(flag_2_not_all_zero)

    ter_1 = Node("ter_1", [], [])

    protocol.add_node(ter_1)

    condition_f_3 = Condition("equal", left="f_3", right=0)
    condition_s_3 = Condition("equal", left="s_3", right="s_2")
    condition_3 = Condition("and", operands=[condition_f_3, condition_s_3])

    flag_2_all_zero = Node("flag_2_all_zero", instructions=["flagged_syndrome"], branches=[Branch(target= "r_3_all_zero", condition= condition_3),
                  Branch(target="r_3_not_all_zero", condition = Condition("not", operand=condition_3))])
    
    protocol.add_node(flag_2_all_zero)

    r_3_not_all_zero = Node("r_3_not_all_zero", ["raw_syndrome"], [Branch(target = "ter_3" )])
    protocol.add_node(r_3_not_all_zero)

    r_3_not_all_zero = Node("r_3_all_zero", ["raw_syndrome"], [Branch(target = "ter_2" )])

    protocol.add_node(r_3_not_all_zero)
    ter_2 = Node("ter_2", [], [])
    protocol.add_node(ter_2)

    

    ter_3 = Node("ter_3", [], [])
    protocol.add_node(ter_3)


    protocol.save_to_file("./protocols/d_5_lai_protocol.json")

    


    return protocol



def build_flag_protocol_chris_d_3():
    n_same =0 
    n_diff = 0 

    protocol = Protocol(start_node="root")

    #the first round has not  flag
    condition_f_1_all_zero = Condition(
    cond_type="equal",
    left="f_1",
    right=0)

    #the first round has a flag
    condition_r_1_flagged = Condition(cond_type = "not", operand= condition_f_1_all_zero)


    #root node
    root_node = Node(
        node_id="root",
        instructions=["flagged_syndrome_1"],
        branches=[Branch(target="r_1_flagged", condition = condition_r_1_flagged),
                  Branch(target="r_1_not_flagged", condition = Condition("not", operand=condition_f_1_all_zero))]

    )
    protocol.add_node(root_node)

    #the first round has a flag 
    r_1_flagged = Node(
        node_id="r_1_flagged",
        instructions=["raw_syndrome"], branches=[Branch(target="ter_1")])
    
    protocol.add_node(r_1_flagged)

    ter_1 = Node("ter_1", [], [])

    protocol.add_node(ter_1)

    #the second round has not flag
    condition_f_2_all_zero = Condition(cond_type="equal", left="f_2", right=0)

    #the second round has a flag
    condition_f_2_flagged = Condition(cond_type = "not", operand= condition_f_2_all_zero)   

    #the synderome of second round is the same as the first round
    condition_s_2_eq_s1 = Condition(cond_type="equal", left="s_2", right="s_1")
    condition_s_2_neq_s1 = Condition(cond_type = "not", operand= condition_s_2_eq_s1)

    r_1_not_flagged  = Node(
        node_id="r_1_not_flagged",
        instructions=["flagged_syndrome_2"],
        branches=[Branch(target="r_2_flagged", condition = condition_f_2_flagged),
                  Branch(target="r_2_not_flagged_and_s2_neq_s1", condition = Condition("and", operands=[condition_f_2_all_zero, condition_s_2_neq_s1])),
                    Branch(target="r_2_not_flagged_and_s2_eq_s1", condition = Condition("and", operands=[condition_f_2_all_zero, condition_s_2_eq_s1]))]
    )
    protocol.add_node(r_1_not_flagged)

    #second round is flagged    
    r_2_flagged = Node(
        node_id="r_2_flagged",
        instructions=["raw_syndrome"],
        branches=[Branch(target="ter_2")]
    )
    protocol.add_node(r_2_flagged)
    ter_2 = Node("ter_2", [], [])
    protocol.add_node(ter_2)
    #second round is not flagged and s_2 != s_1
    r_2_not_flagged_s2_neq_s1 = Node(
        node_id="r_2_not_flagged_and_s2_neq_s1",
        instructions=["raw_syndrome"],
        branches=[Branch(target="ter_3")]
    )
    protocol.add_node(r_2_not_flagged_s2_neq_s1)
    ter_3 = Node("ter_3", [], [])
    protocol.add_node(ter_3)

    #second round is not flagged and s_2 == s_1
    r_2_not_flagged_s2_eq_s1 = Node(
        node_id="r_2_not_flagged_and_s2_eq_s1",
        instructions=["raw_syndrome"],
        branches=[Branch(target="ter_4")]
    )
    protocol.add_node(r_2_not_flagged_s2_eq_s1)
    ter_4 = Node("ter_4", [], [])
    protocol.add_node(ter_4)    

    # Add nodes and branches as per the flag protocol structure
    # This is a placeholder for the actual implementation

    protocol.save_to_file("./out/flag_protocol.json")
    return protocol