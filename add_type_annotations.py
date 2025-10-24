import json
import ast
from typing import get_origin, get_args

class AddAnnotatedTransformer(ast.NodeTransformer):
    def __init__(self, annotations_map):
        self.annotations_map = annotations_map
        self.class_name = None

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        
        # Handle __init__ special case for class methods
        if self.class_name and node.name == '__init__':
            if self.class_name in self.annotations_map and node.name in self.annotations_map[self.class_name]:
                self._add_annotations(node, self.annotations_map[self.class_name][node.name], class_method=True)
            return node

        if self.class_name and self.class_name in self.annotations_map and node.name in self.annotations_map[self.class_name]:
            self._add_annotations(node, self.annotations_map[self.class_name][node.name], class_method=True)
        elif node.name in self.annotations_map:
            self._add_annotations(node, self.annotations_map[node.name])
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        old_class_name = self.class_name
        self.class_name = node.name
        self.generic_visit(node)
        self.class_name = old_class_name
        return node
    
    def _add_annotations(self, func_node, annotations, class_method=False):
        # Add return type annotation
        if func_node.returns is None and "return" in annotations:
            func_node.returns = ast.parse(annotations["return"]).body[0].value

        # Add argument type annotations
        for arg in func_node.args.args:
            if class_method and arg.arg == "self":
                continue
            if arg.arg in annotations and arg.annotation is None:
                arg.annotation = ast.parse(annotations[arg.arg]).body[0].value
        
        for arg in func_node.args.kwonlyargs:
            if arg.arg in annotations and arg.annotation is None:
                arg.annotation = ast.parse(annotations[arg.arg]).body[0].value
        
        for arg in func_node.args.posonlyargs:
            if arg.arg in annotations and arg.annotation is None:
                arg.annotation = ast.parse(annotations[arg.arg]).body[0].value

        if func_node.args.vararg and func_node.args.vararg.arg in annotations and func_node.args.vararg.annotation is None:
            func_node.args.vararg.annotation = ast.parse(annotations[func_node.args.vararg.arg]).body[0].value

        if func_node.args.kwarg and func_node.args.kwarg.arg in annotations and func_node.args.kwarg.annotation is None:
            func_node.args.kwarg.annotation = ast.parse(annotations[func_node.args.kwarg.arg]).body[0].value


def process_code_with_annotations(source_code, add_annotated_import=False):
    annotations_map = {
        "calculate_return": {
            "episode": "Annotated[List[Tuple[int, int, float]], '回合序列，包含(状态，动作，奖励)元组']",
            "gamma": "Annotated[float, '折扣因子']",
            "return": "Annotated[float, '折扣回报']"
        },
        "mc_policy_evaluation": {
            "env": "Annotated[GridWorldMC, '网格世界环境实例']",
            "policy": "Annotated[np.ndarray, '当前策略 (状态空间大小 x 动作空间大小)']",
            "Q": "Annotated[np.ndarray, '动作价值函数 Q(s,a)']",
            "gamma": "Annotated[float, '折扣因子']",
            "num_episodes": "Annotated[int, '每个状态-动作对采样的回合数']",
            "return": "Annotated[np.ndarray, '更新后的动作价值函数 Q(s,a)']"
        },
        "greedy_policy_improvement": {
            "Q": "Annotated[np.ndarray, '动作价值函数 Q(s,a)']",
            "return": "Annotated[np.ndarray, '改进后的贪婪策略']"
        },
        "get_policy_matrix": {
            "policy": "Annotated[np.ndarray, '策略 (状态空间大小 x 动作空间大小)']",
            "grid_size": "Annotated[int, '网格环境的大小']",
            "return": "Annotated[np.ndarray, '用于可视化的策略箭头矩阵']"
        },
        "render_mc_basic_to_array": {
            "rewards": "Annotated[np.ndarray, '奖励矩阵']",
            "Q": "Annotated[np.ndarray, '动作价值函数 Q(s,a)']",
            "policy_arrows": "Annotated[np.ndarray, '策略箭头矩阵']",
            "iteration": "Annotated[int, '当前迭代次数']",
            "return": "Annotated[np.ndarray, '渲染结果的RGB图像数组']"
        },
        "run_mc_basic": {
            "gif_filename": "Annotated[str, '生成的GIF文件名']",
            "fps": "Annotated[int, 'GIF的帧率']",
            "return": "Annotated[Tuple[np.ndarray, np.ndarray, int], '最终的Q值, 策略和迭代次数']"
        },
        "GridWorldMC": {
            "__init__": {
                "size": "Annotated[int, '网格世界的大小']",
                "rewards": "Annotated[Optional[np.ndarray], '自定义奖励矩阵']"
            },
            "_pos_to_state": {
                "pos": "Annotated[np.ndarray, '二维位置坐标 [row, col]']",
                "return": "Annotated[int, '一维状态索引']"
            },
            "_state_to_pos": {
                "state": "Annotated[int, '一维状态索引']",
                "return": "Annotated[np.ndarray, '二维位置坐标 [row, col]']"
            },
            "reset": {
                "start_pos": "Annotated[Optional[np.ndarray], '智能体起始位置']",
                "return": "Annotated[Tuple[int, Dict], '初始状态和信息']"
            },
            "step": {
                "action": "Annotated[int, '采取的动作']",
                "return": "Annotated[Tuple[int, float, bool, bool, Dict], '新状态, 奖励, 是否终止, 是否截断, 信息']"
            },
            "generate_episode": {
                "start_state": "Annotated[int, '回合起始状态']",
                "start_action": "Annotated[int, '回合起始动作']",
                "policy": "Annotated[np.ndarray, '当前策略']",
                "max_steps": "Annotated[int, '回合最大步数']",
                "return": "Annotated[List[Tuple[int, int, float]], '生成的回合序列']"
            }
        }
    }

    tree = ast.parse(source_code)
    
    # Remove redundant inner _pos_to_state definitions
    class RemoveInnerFunc(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # No need for this hack if the definitions are in the correct class
            # This is specifically for redundant inner functions
            # In get_policy_matrix, _pos_to_state is defined for the environment instance
            # In render_mc_basic_to_array, _state_to_pos is defined for the environment instance
            # These are redundant if GridWorldMC instance is passed correctly
            if node.name == "get_policy_matrix":
                node.body = [item for item in node.body if not (isinstance(item, ast.FunctionDef) and item.name == "_pos_to_state")]
            if node.name == "render_mc_basic_to_array":
                node.body = [item for item in node.body if not (isinstance(item, ast.FunctionDef) and item.name == "_state_to_pos")]
            return node
    
    tree = RemoveInnerFunc().visit(tree)

    # Add Annotated import at the correct place
    if add_annotated_import:
        annotated_imported = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                if not any(alias.name == 'Annotated' for alias in node.names):
                    node.names.append(ast.alias(name='Annotated'))
                annotated_imported = True
                break
        if not annotated_imported:
            tree.body.insert(0, ast.parse("from typing import Annotated").body[0])

    transformer = AddAnnotatedTransformer(annotations_map)
    tree = transformer.visit(tree)
    
    return ast.unparse(tree)

def process_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Flag to ensure Annotated is imported only once
    annotated_imported_global = False

    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_code_lines = cell['source']
            if not source_code_lines:
                continue

            source_code = "".join(source_code_lines)

            insert_annotated_in_this_cell = False
            if not annotated_imported_global:
                if "from typing import" in source_code or "import typing" in source_code:
                    insert_annotated_in_this_cell = True
                    annotated_imported_global = True
                elif i == 0: # If it's the very first cell and no typing import, add it
                    insert_annotated_in_this_cell = True
                    annotated_imported_global = True
            
            try:
                modified_code = process_code_with_annotations(source_code, add_annotated_import=insert_annotated_in_this_cell)
                # Ensure each line ends with a newline character, and handle the last line
                cell['source'] = [line + "\n" for line in modified_code.splitlines()]
                # Remove trailing newline from the last line if it's the only one or not needed
                if cell['source'] and not modified_code.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                
                # If there's only one line, and it was empty, ensure it's not made empty
                if not cell['source'] and len(modified_code.strip()) > 0:
                    cell['source'] = [modified_code.strip() + '\n']

            except Exception as e:
                print(f"Error processing code cell: {e}")
                print(f"Original code:\n{source_code}")
                # Keep original code if processing fails
                pass 
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    process_notebook('第3章补充资料/MC_Basic.ipynb', '第3章补充资料/MC_Basic_annotated.ipynb')
