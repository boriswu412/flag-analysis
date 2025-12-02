import json
import os
from typing import Optional, List, Dict

class Condition:
    def __init__(self, cond_type: str, left=None, right=None, operand=None, operands=None):
        self.type = cond_type
        self.left = left          # For binary conditions
        self.right = right        # For binary conditions
        self.operand = operand    # For unary conditions (like 'not')
        self.operands = operands  # For n-ary conditions (like 'and', 'or')

    def to_dict(self):
        d = {"type": self.type}
        if self.type in ("equal", "not_equal",
                        "greater", "less", "greater_equal", "less_equal"):
            d["left"] = self.left
            d["right"] = self.right

        elif self.type == "not":
            d["operand"] = self.operand.to_dict()

        elif self.type in ("and", "or"):
            d["operands"] = [op.to_dict() for op in self.operands]

        else:
            raise ValueError(f"Unknown condition type: {self.type}")

        return d


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
        instructions=["flagged_syndrome_1"],
        branches=[Branch(target="all_zero", condition= condition_1),
                  Branch(target="not_all_zero", condition = Condition("not", operand=condition_1))]
    )

    protocol.add_node(root_node)


    #node all zero in first round
    round_1_all_zero = Node("all_zero",[], [])

    protocol.add_node(round_1_all_zero)
    

    condition_2 = Condition("equal", left="f_2", right=0)
    #node not all zero in first round
    round_1_not_all_zero = Node(node_id="not_all_zero" ,instructions=["flagged_syndrome_2"],branches=[Branch(target="flag_2_all_zero", condition = condition_2),
                  Branch(target = "flag_2_not_all_zero", condition = Condition("not", operand=condition_2))])
    
    protocol.add_node(round_1_not_all_zero)

    flag_2_not_all_zero = Node("flag_2_not_all_zero", ["raw_syndrome"], [Branch(target = "ter_1" )])
    protocol.add_node(flag_2_not_all_zero)

    ter_1 = Node("ter_1", [], [])

    protocol.add_node(ter_1)

    condition_f_3 = Condition("equal", left="f_3", right=0)
    condition_s_3 = Condition("equal", left="s_3", right="s_2")
    condition_3 = Condition("and", operands=[condition_f_3, condition_s_3])

    flag_2_all_zero = Node("flag_2_all_zero", instructions=["flagged_syndrome_3"], branches=[Branch(target= "r_3_all_zero", condition= condition_3),
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