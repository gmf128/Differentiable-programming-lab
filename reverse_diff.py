import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random
import irvisitor

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    def get_lhs_unique_id(lhs):
        """
        A utility function to get lhs`s ID to judge if overwriting happens
        Input : x => Output: 'x'
        Input : x[0] => Output: 'x[0]'
        Input : x.member => Output: 'x.member'

        Parameters
        ----------
        lhs
        adjoint_id_dict: the mapping from var id to _dvar id

        Returns
        -------

        """
        if isinstance(lhs, loma_ir.Var):
            id = lhs.id
            return id
        elif isinstance(lhs, loma_ir.ArrayAccess):
            # get id
            indexes = []
            ptr = 0
            child = lhs.array
            indexes.append(lhs.index)
            while isinstance(child, loma_ir.ArrayAccess):
                indexes.append(child.index)
                ptr += 1
                child = child.array
                t = child.t
            id = child.id

            while ptr >= 0:
                id += '['
                id += str(indexes[ptr])
                id += ']'
                ptr -= 1
            return id

        elif isinstance(lhs, loma_ir.StructAccess):
            # struct.x.y.z => dstruct.x.y.z
            # get id
            members = []
            ptr = 0
            child = lhs.struct
            members.append(lhs.member_id)
            while isinstance(child, loma_ir.StructAccess):
                members.append(child.member_id)
                ptr += 1
                child = child.struct
            id = child.id

            while ptr >= 0:
                id += '.'
                id += str(members[ptr])
                ptr -= 1

            return id

    def get_lhs_adjoint_var(lhs, adjoint_id_dict):
        """
        A utility function to get adjoint variables from lhs.
        Input : x => Output: _dx
        Input : x[0] => Output: _dx[0]
        Input : x.member => Output: _dx.member

        Parameters
        ----------
        lhs
        adjoint_id_dict: the mapping from var id to _dvar id

        Returns
        -------

        """
        if isinstance(lhs, loma_ir.Var):
            id = lhs.id
            return loma_ir.Var(adjoint_id_dict[id], t=lhs.t)
        elif isinstance(lhs, loma_ir.ArrayAccess):
            # get id
            indexes = []
            ptr = 0
            child = lhs.array
            indexes.append(lhs.index)
            t = lhs.t
            while isinstance(child, loma_ir.ArrayAccess):
                indexes.append(child.index)
                ptr += 1
                child = child.array
                t = child.t
            id = child.id
            if id in adjoint_id_dict.keys():
                # which means the variable requires grad
                d_array_id = adjoint_id_dict[id]
                d_array_access = loma_ir.Var(d_array_id, t=t)
                while ptr >= 0:
                    d_array_access = loma_ir.ArrayAccess(d_array_access, indexes[ptr])
                    ptr -= 1
            return d_array_access

        elif isinstance(lhs, loma_ir.StructAccess):
            # struct.x.y.z => dstruct.x.y.z
            # get id
            members = []
            types = []
            ptr = 0
            child = lhs.struct
            members.append(lhs.member_id)
            types.append(lhs.t)
            while isinstance(child, loma_ir.StructAccess):
                members.append(child.member_id)
                ptr += 1
                child = child.struct
                types.append(child.t)
            id = child.id
            if id in adjoint_id_dict.keys():
                # which means the variable requires grad
                d_struct_id = adjoint_id_dict[id]
                d_struct_access = loma_ir.Var(d_struct_id)
                while ptr >= 0:
                    d_struct_access = loma_ir.StructAccess(d_struct_access, members[ptr], t=types[ptr])
                    ptr -= 1

            return d_struct_access

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(
                call,
                lineno=node.lineno)]

        def mutate_call(self, node):
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t=arg.t)
                    try:
                        self.tmp_declare_stmts.append(loma_ir.Declare(
                            tmp_name, arg.t))
                    except Exception as e:
                        print(node)
                    self.tmp_assign_stmts.append(loma_ir.Assign(
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    # Apply the differentiation.
    class LoopTreeNode:
        def __init__(self, depth, max_iter):
            self.depth = depth
            self.children = []
            self.children_index = -1
            self.lineage_index = []
            self.max_iter = max_iter
            self.total_max_iter = max_iter

        def add_child(self, node):
            self.children.append(node)
            self.children_index += 1
            l = self.lineage_index.copy()
            l.append(self.children_index)
            node.lineage_index = l
            node.total_max_iter *= self.total_max_iter

        def get_children_cnt(self):
            return self.children_index

        def get_lineage_index(self):
            index = ""
            for ind in self.lineage_index:
                index += str(ind)
                index += "_"
            return index

    class LoopTree:
        def __init__(self):
            self.root = LoopTreeNode(-1, 1)
            self.current = self.root
            self.cur_depth = -1

        def declare_all(self):
            return self.declare(self.root)

        def declare(self, node):
            res = []
            if node.depth == -1:
                for i in range(0, node.children_index + 1):
                    loop_var_id = "_loop_counter_"
                    loop_var_id += node.get_lineage_index()
                    loop_var_id += str(i)
                    res.append(loma_ir.Declare(loop_var_id, loma_ir.Int()))
                    # recursive
                    res += irmutator.flatten(self.declare(node.children[i]))
            else:
                for i in range(0, node.children_index + 1):
                    loop_var_id = "_loop_counter_"
                    loop_var_id += node.get_lineage_index()
                    loop_var_id += str(i)
                    res.append(loma_ir.Declare(loop_var_id, loma_ir.Array(t=loma_ir.Int(), static_size=node.total_max_iter)))
                    loop_var_id_tmp = loop_var_id + "_tmp"
                    loop_var_id_ptr = loop_var_id + "_ptr"
                    res.append(loma_ir.Declare(loop_var_id_tmp, loma_ir.Int()))
                    res.append(loma_ir.Declare(loop_var_id_ptr, loma_ir.Int()))
                    # recursive
                    res += irmutator.flatten(self.declare(node.children[i]))
            return res

        def reset(self):
            self.current = self.root
            self.cur_depth = -1


    class PrimalBuildMutator(irmutator.IRMutator):
        def __init__(self):
            super().__init__()
            self.overwrite_dict = {}
            self.adjoint_id_dict = {}
            self.output_args = []
            self.tmp_adjoint_var_names = []
            self.tmp_adjoint_var_cnt = 0
            self.tmp_adjoint_var_decl_code = []
            self.loopTree = LoopTree()
        def mutate_function_def(self, node):
            for arg in node.args:
                if arg.i == loma_ir.Out():
                    self.output_args.append(arg.id)
            primary_code = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            primary_code = irmutator.flatten(primary_code)
            return primary_code
        def mutate_declare(self, node):
            # First, we need to generate z as primal code, and declare the adjoint value _dz
            res = []
            res.append(node)  # primal
            # do not add _d Var if the type is Int
            #TODO
            if isinstance(node.t, loma_ir.Int):
                return res
            id = node.target
            adjoint_id = f"_d{id}_{random_id_generator()}"
            self.adjoint_id_dict[id] = adjoint_id
            res.append(loma_ir.Declare(adjoint_id, t=node.t))
            return res

        def mutate_call_stmt(self, node):
            res = []
            # Deal with side effects
            # For example: def foo(x: In[float], y:Out[float]) ...
            # foo(x, y): indicates that the value of y will be changed after call stmt, so we have to use stack to store the previous value of y
            assert isinstance(node.call, loma_ir.Call)
            call = node.call
            func_id = call.id
            # Get arg types IN or OUT
            primary_func_def = funcs[func_id]
            assert isinstance(primary_func_def, loma_ir.FunctionDef)
            new_primary_args = []
            for i in range(len(call.args)):
                arg = call.args[i]
                arg_def = primary_func_def.args[i]
                if arg_def.i == loma_ir.Out():
                    if check_lhs_is_output_arg(arg, self.output_args):
                        return []
                    else:
                        # PUSH into the stack
                        # stack name: _tmp_stack_{typename}
                        # stack ptr: _stack_ptr_{typename}
                        target_type = self.get_left_expression_type(arg)
                        type_name = type_to_string(target_type)
                        stack_name = f"_tmp_stack_{type_name}"
                        stack_ptr_name = f"_stack_ptr_{type_name}"
                        if type_name in self.overwrite_dict.keys():
                            self.overwrite_dict[type_name]['count'] += self.loopTree.current.total_max_iter
                        else:
                            self.overwrite_dict[type_name] = {'type': target_type, 'count': self.loopTree.current.total_max_iter}
                        stack_access = loma_ir.ArrayAccess(loma_ir.Var(stack_name), loma_ir.Var(stack_ptr_name),
                                                           t=target_type)
                        # Push the target into the stack
                        res.append(loma_ir.Assign(stack_access, arg))
                        # Advance stack pointer
                        res.append(loma_ir.Assign(loma_ir.Var(stack_ptr_name),
                                                  loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(stack_ptr_name),
                                                                   loma_ir.ConstInt(1))))
                        # Assist to declare tmp adjoint variable used in Adjoint Period
                        if type_name != "int":
                            tmp_var_name = f"_tmp_adjoint_var_{self.tmp_adjoint_var_cnt}_{random_id_generator()}"
                            self.tmp_adjoint_var_names.append(tmp_var_name)
                            self.tmp_adjoint_var_decl_code.append(loma_ir.Declare(tmp_var_name, t=target_type))
                            self.tmp_adjoint_var_cnt += 1

            # Call the primary function
            res.append(node)
            return res
        def mutate_stmt(self, node):
            match node:
                case loma_ir.Return():
                    return []
                case loma_ir.Declare():
                    return self.mutate_declare(node)
                case loma_ir.Assign():
                    return self.mutate_assign(node)
                case loma_ir.IfElse():
                    return self.mutate_ifelse(node)
                case loma_ir.CallStmt():
                    return self.mutate_call_stmt(node)
                case loma_ir.While():
                    return self.mutate_while(node)
                case _:
                    assert False, f'Visitor error: unhandled statement {node}'

        def mutate_while(self, node):
            # In primary mode, max_iter and cond have no change
            res = []
            new_body = []

            # update loopTree
            parent = self.loopTree.current
            self.loopTree.cur_depth += 1
            this_node = LoopTreeNode(self.loopTree.cur_depth, node.max_iter)
            self.loopTree.current.add_child(this_node)
            self.loopTree.current = this_node
            loop_var_id = "_loop_counter_"
            tmp_name = parent.get_lineage_index() + str(parent.children_index)
            loop_var_id += tmp_name if self.loopTree.cur_depth == 0 else (tmp_name + '_tmp')
            children_cnt = -1
            for stmt in node.body:
                if isinstance(stmt, loma_ir.While):
                    # Initiate tmp loop var
                    children_cnt += 1
                    loop_child_var_tmp_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}_tmp"
                    loop_child_var_stack_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}"
                    loop_child_var_ptr_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}_ptr"
                    new_body.append(loma_ir.Assign(loma_ir.Var(loop_child_var_tmp_id), loma_ir.ConstInt(0)))
                    # Recursion
                    new_body.append(self.mutate_while(stmt))
                    # Record
                    new_body.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var(loop_child_var_stack_id), index=loma_ir.Var(loop_child_var_ptr_id)), loma_ir.Var(loop_child_var_tmp_id)))
                    new_body.append(loma_ir.Assign(loma_ir.Var(loop_child_var_ptr_id), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(loop_child_var_ptr_id), loma_ir.ConstInt(1))))
                else:
                    new_body.append(self.mutate_stmt(stmt))
            new_body.append(loma_ir.Assign(loma_ir.Var(loop_var_id), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(loop_var_id), loma_ir.ConstInt(1))))
            new_body = irmutator.flatten(new_body)
            res.append(loma_ir.While(cond=node.cond, max_iter=node.max_iter, body=new_body))

            # return to parent node
            self.loopTree.current = parent
            self.loopTree.cur_depth -= 1

            return res

        def mutate_assign(self, node):
            # In reverse mode, assign is a rather hard stmt to implement, due to the **side effect**, which means the value of variable is changed
            # Example: def f(x, y){ z; z = x + y; z = z * x + z * y; return z}
            # For the sake of clarity, we first number these same zs: def f(x, y){ z0; z0 = x + y; z1 = z0 * x + z0 * y; return z1}
            # In reverse mode, we will update the adjoints: {dz1 += dreturn; dx += z0 * dz1; dz0 += x * dz1; dy += z0 * dz1; dz0 += y * dz1; dx += dz0; dy += dz0;}
            # If we don`t distinguish the same variable, then we need to use tmp variable to avoid bugs caused by overwritting
            # {
            #   z; dz; stack; stack.push(z); z = x + y; stack.push(z); z = z * x + z * y; return z; //primary
            #   dz += dreturn; | tmp1 = 0; z = stack.pop(); dx += z * dz; tmp1 += x * dz; dy += z * dz; tmp1+= y * dz; dz = tmp1; | dx += dz; dy += dz;
            #       }

            # Stack manipulation happens only when overwriting; Stack size is equal to the number of assignment statements(every assignment stmt is overwriting, otherwise it is a declare stmt)
            # Push the old value into stack before assignment statement in Primary Period
            # and pop the value before the assignment in Adjoint Period
            res = []
            # deal with output variable
            if check_lhs_is_output_arg(node.target, self.output_args):
                return []
            # stack name: _tmp_stack_{typename}
            # stack ptr: _stack_ptr_{typename}
            target_type = self.get_left_expression_type(node.target)
            type_name = type_to_string(target_type)
            stack_name = f"_tmp_stack_{type_name}"
            stack_ptr_name = f"_stack_ptr_{type_name}"
            if type_name in self.overwrite_dict.keys():
                self.overwrite_dict[type_name]['count'] += self.loopTree.current.total_max_iter
            else:
                self.overwrite_dict[type_name] = {'type': target_type, 'count': self.loopTree.current.total_max_iter}
            stack_access = loma_ir.ArrayAccess(loma_ir.Var(stack_name), loma_ir.Var(stack_ptr_name), t=target_type)
            # Push the target into the stack
            res.append(loma_ir.Assign(stack_access, node.target))
            # Primary code
            res.append(node)
            # Advance stack pointer
            res.append(loma_ir.Assign(loma_ir.Var(stack_ptr_name),
                                      loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(stack_ptr_name), loma_ir.ConstInt(1))))
            # Assist to declare tmp adjoint variable used in Adjoint Period
            if type_name != "int":
                tmp_var_name = f"_tmp_adjoint_var_{self.tmp_adjoint_var_cnt}_{random_id_generator()}"
                self.tmp_adjoint_var_names.append(tmp_var_name)
                self.tmp_adjoint_var_decl_code.append(loma_ir.Declare(tmp_var_name, t=target_type))
                self.tmp_adjoint_var_cnt += 1
            return res

        def get_left_expression_type(self, node):
            match node:
                case loma_ir.Var():
                    return node.t
                case loma_ir.ArrayAccess():
                    return node.t
                case loma_ir.StructAccess():
                    return node.t
                case loma_ir.ConstFloat():
                    raise AssertionError("Constant cannot be lvalue")
                case loma_ir.ConstInt():
                    raise AssertionError("Constant cannot be lvalue")
                case loma_ir.BinaryOp():
                    raise AssertionError("Binary Operation cannot be lvalue")
                case loma_ir.Call():
                    raise AssertionError("Function call cannot be lvalue")
                case _:
                    assert False, f'Visitor error: unhandled expression {node}'
        def mutate_expr(self, node):
            return super().mutate_expr(node)

        def mutate_ifelse(self, node):
            new_cond = self.mutate_expr(node.cond)
            new_then_stmts = [self.mutate_stmt(stmt) for stmt in node.then_stmts]
            new_else_stmts = [self.mutate_stmt(stmt) for stmt in node.else_stmts]
            # Important: mutate_stmt can return a list of statements. We need to flatten the lists.
            new_then_stmts = irmutator.flatten(new_then_stmts)
            new_else_stmts = irmutator.flatten(new_else_stmts)
            return loma_ir.IfElse( \
                new_cond,
                new_then_stmts,
                new_else_stmts,
                lineno=node.lineno)

    class RevDiffMutator(irmutator.IRMutator):
        def __init__(self):
            super().__init__()
            self.adjoint_id_dict = {}
            self.overwrite_id = None
            self.tmp_adjoint_var_names = None
            self.overwrite_cnt = 0
            self.primary_mutator = None
            self.output_args = []
            self.loopTree = None
        def mutate_function_def(self, node):
            # HW2:
            # FunctionDef(string id, arg* args, stmt* body, bool is_simd, type? ret_type)

            # First, do function call normalization(append tmp variables, see Homework3 to know why)
            call_normalize_mutator = CallNormalizeMutator()
            node = call_normalize_mutator.mutate_function_def(node)

            new_args = []
            adjoint_id_dict = {}
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    adjoint_id = f"_d{arg.id}_{random_id_generator()}"
                    adjoint_arg = loma_ir.Arg(id=adjoint_id, t=arg.t, i=loma_ir.Out())
                    adjoint_id_dict[arg.id] = adjoint_id
                    new_args.append(adjoint_arg)
                elif arg.i == loma_ir.Out():
                    # refs out
                    # Important: No need for new_args.append(arg), we dont need a output variable in reverse mode anymore!
                    adjoint_id = f"_d{arg.id}_{random_id_generator()}"
                    adjoint_arg = loma_ir.Arg(id=adjoint_id, t=arg.t, i=loma_ir.In())
                    adjoint_id_dict[arg.id] = adjoint_id
                    new_args.append(adjoint_arg)
            if node.ret_type == loma_ir.Float():
                d_return_id = f"_dreturn_{random_id_generator()}"
                d_return = loma_ir.Arg(id=d_return_id, t=loma_ir.Float(), i=loma_ir.In())
                new_args.append(d_return)
                self.adjoint = loma_ir.Var(d_return_id)
            elif node.ret_type == None:
                pass
            elif node.ret_type == loma_ir.Int():
                d_return_id = f"_dreturn_{random_id_generator()}"
                d_return = loma_ir.Arg(id=d_return_id, t=loma_ir.Float(), i=loma_ir.In())
                new_args.append(d_return)
                self.adjoint = loma_ir.Var(d_return_id)
            elif isinstance(node.ret_type, loma_ir.Struct):
                d_return_id = f"_dreturn_{random_id_generator()}"
                d_return = loma_ir.Arg(id=d_return_id, t=node.ret_type, i=loma_ir.In())
                new_args.append(d_return)
                self.adjoint = loma_ir.Var(d_return_id)
            else:
                raise NotImplementedError("Function ret type which is not float has not been implemented yet")
            self.adjoint_id_dict = adjoint_id_dict
            new_args = tuple(new_args)
            # Primary mode: Declare intermediate variables
            primary_mutator = PrimalBuildMutator()
            self.primary_mutator = primary_mutator
            primary_code = primary_mutator.mutate_function_def(node)
            declare_dict = primary_mutator.adjoint_id_dict
            self.output_args = primary_mutator.output_args
            for key in declare_dict.keys():
                self.adjoint_id_dict[key] = declare_dict[key]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.

            # Declare the overwriting stack
            overwrite_dict = primary_mutator.overwrite_dict
            stack_declare_code = []
            # Tmp variables have been generated in Primary Period
            self.tmp_adjoint_var_names = primary_mutator.tmp_adjoint_var_names
            self.tmp_adjoint_var_names.reverse()
            tmp_adjoint_var_declare_code = primary_mutator.tmp_adjoint_var_decl_code
            for key in overwrite_dict.keys():
                type = overwrite_dict[key]['type']
                declare_stmt = loma_ir.Declare(f"_tmp_stack_{key}", \
                                               t=loma_ir.Array(t=type,
                                                               static_size=overwrite_dict[key]['count']))
                declare_ptr_stmt = loma_ir.Declare(f"_stack_ptr_{key}", t=loma_ir.Int())
                stack_declare_code.append(declare_stmt)
                stack_declare_code.append(declare_ptr_stmt)
                # Declare the tmp variables to store _dz
            self.loopTree = primary_mutator.loopTree
            loop_var_declare_code = primary_mutator.loopTree.declare_all()
            self.loopTree.reset()
            # Reverse mode: We have to visit the stmts reversely
            new_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]

            new_body = stack_declare_code + loop_var_declare_code + primary_code \
                       + tmp_adjoint_var_declare_code + irmutator.flatten(new_body)

            self.overwrite_cnt = 0
            new_ret_type = None
            rev_func_id = func_to_rev[node.id]
            lineno = funcs[rev_func_id].lineno

            return loma_ir.FunctionDef( \
                diff_func_id,
                new_args,
                new_body,
                node.is_simd,
                new_ret_type,
                lineno=lineno)

        def mutate_return(self, node):
            # HW2:
            # mutate_return should back-propagate d_return to the expr in return statement
            assert self.adjoint is not None
            return self.mutate_expr(node.val)

        def mutate_declare(self, node):
            # HW2:
            # Declare means we have a new variable now, which is a new node in the computing graph
            #
            #   x <---
            #         |-- 'f' <-- _dz ---(z=f(x, y))
            #   y <---
            if not isinstance(node.t, loma_ir.Int):
                self.adjoint = loma_ir.Var(self.adjoint_id_dict[node.target])
            if node.val is None:
                return []
            return self.mutate_expr(node.val)

        def mutate_assign(self, node):
            # HW2:
            # In reverse mode, assign is a rather hard stmt to implement, due to the **side effect**, which means the value of variable is changed
            # Example: def f(x, y){ z; z = x + y; z = z * x + z * y; return z}
            # For the sake of clarity, we first number these same zs: def f(x, y){ z0; z0 = x + y; z1 = z0 * x + z0 * y; return z1}
            # In reverse mode, we will update the adjoints: {dz1 += dreturn; dx += z0 * dz1; dz0 += x * dz1; dy += z0 * dz1; dz0 += y * dz1; dx += dz0; dy += dz0;}
            # If we don`t distinguish the same variable, then we need to use tmp variable to avoid bugs caused by overwritting
            # {
            #   z; dz; stack; stack.push(z); z = x + y; stack.push(z); z = z * x + z * y; return z; //primary
            #   dz += dreturn; | tmp1 = 0; z = stack.pop(); dx += z * dz; tmp1 += x * dz; dy += z * dz; tmp1+= y * dz; dz = tmp1; | dx += dz; dy += dz;
            #       }

            # Stack manipulation happens only when overwriting; Stack size is equal to the number of assignment statements(every assignment stmt is overwriting, otherwise it is a declare stmt)
            # Push the old value into stack before assignment statement in Primary Period
            # and pop the value before the assignment in Adjoint Period
            res = []
            # deal with output variable
            if check_lhs_is_output_arg(node.target, self.output_args):
                self.adjoint = get_lhs_adjoint_var(node.target, self.adjoint_id_dict)
                res += self.mutate_expr(node.val)
                return res
            # Pop the old value
            target_type = self.primary_mutator.get_left_expression_type(node.target)
            id = type_to_string(target_type)
            # Don`t forget to update stack ptr
            res.append(loma_ir.Assign(loma_ir.Var(f"_stack_ptr_{id}"),
                                      loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(f"_stack_ptr_{id}"),
                                                       loma_ir.ConstInt(1))))
            stack_pop_stmt = loma_ir.ArrayAccess(loma_ir.Var(f"_tmp_stack_{id}"), loma_ir.Var(f"_stack_ptr_{id}"))
            res.append(loma_ir.Assign(target=node.target, val=stack_pop_stmt))

            # back_propagate carefully
            # if node.target is int, then no need for grad backpropagation
            if isinstance(node.target.t, loma_ir.Int):
                return res

            self.adjoint = get_lhs_adjoint_var(node.target, self.adjoint_id_dict)

            # get overwrite_id
            self.overwrite_id = get_lhs_unique_id(node.target)
            res += self.mutate_expr(node.val)

            # Update adjoint
            res += accum_deriv(get_lhs_adjoint_var(node.target, self.adjoint_id_dict), loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]), overwrite=True)
            if self.loopTree.cur_depth != -1:
                if isinstance(target_type, loma_ir.Int):
                    res.append(loma_ir.Assign(loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]), loma_ir.ConstInt(0)))
                else:
                    res.append(loma_ir.Assign(loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]),
                                              loma_ir.ConstFloat(0.0)))
            self.overwrite_cnt += 1
            # Destroy the overwrite_id class val since it should not have a value when the stmt is not Assign
            self.overwrite_id = None

            return res

        def mutate_ifelse(self, node):
            new_else_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            new_then_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            # Important: mutate_stmt can return a list of statements. We need to flatten the lists.
            new_then_stmts = irmutator.flatten(new_then_stmts)
            new_else_stmts = irmutator.flatten(new_else_stmts)
            return [loma_ir.IfElse( \
                node.cond,
                new_then_stmts,
                new_else_stmts,
                lineno=node.lineno)]

        def mutate_call_stmt(self, node):
            # HW3:
            res = []
            call = node.call
            func_id = call.id
            # Get arg types IN or OUT
            primary_func_def = funcs[func_id]
            assert isinstance(primary_func_def, loma_ir.FunctionDef)
            diff_args = []
            out_args = []
            n_args = len(node.args)
            for i in range(n_args):
                arg = call.args[i]
                arg_def = primary_func_def.args[i]
                if arg_def.i == loma_ir.In():
                    diff_args.append(arg)
                    diff_args.append(get_lhs_adjoint_var(arg, self.adjoint_id_dict))
                else:
                    diff_args.append(get_lhs_adjoint_var(arg, self.adjoint_id_dict))
                    # Pop the old value
                    if check_lhs_is_output_arg(arg, self.output_args):
                        continue
                    target_type = self.primary_mutator.get_left_expression_type(arg)
                    id = type_to_string(target_type)
                    # Don`t forget to update stack ptr
                    res.append(loma_ir.Assign(loma_ir.Var(f"_stack_ptr_{id}"),
                                              loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(f"_stack_ptr_{id}"),
                                                               loma_ir.ConstInt(1))))
                    stack_pop_stmt = loma_ir.ArrayAccess(loma_ir.Var(f"_tmp_stack_{id}"),
                                                         loma_ir.Var(f"_stack_ptr_{id}"))
                    res.append(loma_ir.Assign(target=arg, val=stack_pop_stmt))
                    out_args.append(arg)

            # Call the differential function
            diff_func_id = f"_d_rev_{func_id}"
            diff_call = loma_ir.Call(diff_func_id, diff_args)
            res.append(loma_ir.CallStmt(diff_call))

            # After call stmt, update _d_var and tmp var
            for arg in out_args:
                res += accum_deriv(get_lhs_adjoint_var(arg, self.adjoint_id_dict),
                                   loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]), overwrite=True)
                self.overwrite_cnt += 1
                self.overwrite_id = None
            return res

        def mutate_while(self, node):
            res = []
            parent = self.loopTree.current
            this_node = self.loopTree.current.children[self.loopTree.current.children_index]
            self.loopTree.cur_depth += 1

            loop_var_id = "_loop_counter_"
            tmp_name = parent.get_lineage_index() + str(self.loopTree.current.children_index)
            loop_var_id += tmp_name if self.loopTree.cur_depth == 0 else (
                        tmp_name + '_tmp')
            new_cond = loma_ir.BinaryOp(loma_ir.Greater(), loma_ir.Var(loop_var_id), loma_ir.ConstInt(0))
            new_body = []
            children_cnt = this_node.children_index + 1

            self.loopTree.current = this_node

            for stmt in reversed(node.body):
                if isinstance(stmt, loma_ir.While):
                    # Initiate tmp loop var
                    children_cnt -= 1
                    loop_child_var_tmp_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}_tmp"
                    loop_child_var_stack_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}"
                    loop_child_var_ptr_id = f"_loop_counter_{this_node.get_lineage_index()}{children_cnt}_ptr"
                    new_body.append(loma_ir.Assign(loma_ir.Var(loop_child_var_ptr_id),
                                                   loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(loop_child_var_ptr_id),
                                                                    loma_ir.ConstInt(1))))
                    new_body.append(loma_ir.Assign(
                                        loma_ir.Var(loop_child_var_tmp_id),
                                        loma_ir.ArrayAccess(loma_ir.Var(loop_child_var_stack_id),
                                                                       index=loma_ir.Var(loop_child_var_ptr_id))))
                    # Recursion
                    new_body.append(self.mutate_while(stmt))
                else:
                    new_body.append(self.mutate_stmt(stmt))

            new_body.append(loma_ir.Assign(loma_ir.Var(loop_var_id),
                                           loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(loop_var_id),
                                                            loma_ir.ConstInt(1))))
            new_body = irmutator.flatten(new_body)
            res.append(loma_ir.While(cond=new_cond, max_iter=node.max_iter, body=new_body))

            self.loopTree.current = parent
            self.loopTree.cur_depth -= 1
            self.loopTree.current.children_index -= 1

            return res

        def mutate_const_float(self, node):
            # HW2:
            # Constants do not require grads, so, mutate_const_** just does nothing.
            return []

        def mutate_const_int(self, node):
            # HW2:
            # Constants do not require grads, so, mutate_const_** just does nothing.
            return []

        def mutate_var(self, node):
            # HW2:
            # When mutate_var is called, it means that adjoint value has been propagated to the variable, and we have to update the d_var
            # dx += adjoint
            if node.id in self.adjoint_id_dict.keys():
                # which means the variable requires grad
                d_var_id = self.adjoint_id_dict[node.id]
                # Deal with overwrite properly:
                test_id = get_lhs_unique_id(node)
                if self.overwrite_id is not None and test_id == self.overwrite_id:
                    d_var_id = self.tmp_adjoint_var_names[self.overwrite_cnt]
                d_var_update_list = accum_deriv(loma_ir.Var(d_var_id, t=node.t), self.adjoint, overwrite=False)
                return d_var_update_list
            else:
                # requires no grad: Int or dispatch
                return []

        def mutate_array_access(self, node):
            # HW2:
            # array[index][index2]..[indexN] => _d_array[index][index][index2]..[indexN]
            # get id
            test_id = get_lhs_unique_id(node)
            if self.overwrite_id is not None and test_id == self.overwrite_id:
                d_var_update = loma_ir.Assign(target=self.tmp_adjoint_var_names[self.overwrite_cnt],
                                              val=loma_ir.BinaryOp(loma_ir.Add(),
                                                                   self.tmp_adjoint_var_names[self.overwrite_cnt],
                                                                   self.adjoint))
                return [d_var_update]
            indexes = []
            ptr = 0
            child = node.array
            indexes.append(node.index)
            while isinstance(child, loma_ir.ArrayAccess):
                indexes.append(child.index)
                ptr += 1
                child = child.array
            id = child.id
            if id in self.adjoint_id_dict.keys():
                # which means the variable requires grad
                d_array_id = self.adjoint_id_dict[id]
                d_array_access = loma_ir.Var(d_array_id)
                while ptr >= 0:
                    d_array_access = loma_ir.ArrayAccess(d_array_access, indexes[ptr])
                    ptr -= 1
                d_var_update = loma_ir.Assign(target=d_array_access,
                                              val=loma_ir.BinaryOp(loma_ir.Add(), d_array_access, self.adjoint))
            else:
                raise AssertionError
            return [d_var_update]

        def mutate_struct_access(self, node):
            # HW2:
            # struct.x.y.z => dstruct.x.y.z
            # get id
            test_id = get_lhs_unique_id(node)
            if self.overwrite_id is not None and test_id == self.overwrite_id:
                d_var_update = loma_ir.Assign(target=loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]),
                                              val=loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(self.tmp_adjoint_var_names[self.overwrite_cnt]), self.adjoint))
                return [d_var_update]
            members = []
            ptr = 0
            child = node.struct
            members.append(node.member_id)
            while isinstance(child, loma_ir.StructAccess):
                members.append(child.member_id)
                ptr += 1
                child = child.struct
            id = child.id
            if id in self.adjoint_id_dict.keys():
                # which means the variable requires grad
                d_struct_id = self.adjoint_id_dict[id]
                d_struct_access = loma_ir.Var(d_struct_id)
                while ptr >= 0:
                    d_struct_access = loma_ir.StructAccess(d_struct_access, members[ptr])
                    ptr -= 1
                d_var_update = loma_ir.Assign(target=d_struct_access,
                                              val=loma_ir.BinaryOp(loma_ir.Add(), d_struct_access, self.adjoint))
            else:
                raise AssertionError
            return [d_var_update]

        def mutate_add(self, node):
            # HW2:
            #   expr1 <-delta--
            #                  |-- '+' <--delta-- res
            #   expr2 <-delta--
            return self.mutate_expr(node.left) + self.mutate_expr(node.right)

        def mutate_sub(self, node):
            # HW2:
            #   expr1 <---delta-----
            #                       |-- '-' <--delta-- res
            #   expr2 <-minus delta--
            res = []
            res += self.mutate_expr(node.left)
            tmp_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), self.adjoint)
            res += self.mutate_expr(node.right)
            self.adjoint = tmp_adjoint
            return res

        def mutate_mul(self, node):
            # HW2:
            #   expr1 <--delta*expr2-
            #                        |-- '*' <--delta-- res
            #   expr2 <--delta*expr1-
            res = []
            tmp_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, node.right)
            res += self.mutate_expr(node.left)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, node.left)
            res += self.mutate_expr(node.right)
            self.adjoint = tmp_adjoint
            return res

        def mutate_div(self, node):
            # HW2:
            #   expr1 <-  delta / expr2---------
            #                                   |-- '/' <--d-- res
            #   expr2 <-(-expr1 / sqr(expr2))*d-
            res = []
            tmp_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), tmp_adjoint, node.right)
            res += self.mutate_expr(node.left)
            y_sqr = loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right)
            frac = loma_ir.BinaryOp(loma_ir.Div(), node.left, y_sqr)
            d_out_d_y = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), frac)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, d_out_d_y)
            res += self.mutate_expr(node.right)
            self.adjoint = tmp_adjoint
            return res

        def mutate_call(self, node):
            # HW2:
            res = []
            tmp_adjoint = self.adjoint
            func_id = node.id
            match func_id:
                case "sin":
                    #   x  <--delta * cos(x)-- 'sin' <--delta-- y
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, loma_ir.Call('cos', node.args))
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "cos":
                    #   x  <--minus delta * sin(x)-- 'cos' <--delta-- y
                    dsin = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, loma_ir.Call('sin', node.args))
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), dsin)
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "sqrt":
                    #   x  <-- delta / (2 * y)-- 'sqrt' <--delta-- y
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), tmp_adjoint,
                                                    loma_ir.BinaryOp(loma_ir.Add(), node, node))
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "exp":
                    #   x  <-- delta * y -- 'exp' <--delta-- y
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint,
                                                    node)
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "log":
                    #   x  <-- delta / x -- 'log' <--delta-- y
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), tmp_adjoint,
                                                    node.args[0])
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "pow":
                    #   x <-  delta * y * x^(y-1)-------
                    #                                   |-- 'pow' <--d-- x^y
                    #   y <-  delta * x^y * log(x) -----
                    x = node.args[0]
                    y = node.args[1]
                    x_pow_y_minus_one = loma_ir.Call('pow', [x, loma_ir.BinaryOp(loma_ir.Sub(), y, loma_ir.ConstInt(1))])
                    partial_x = loma_ir.BinaryOp(loma_ir.Mul(), y, x_pow_y_minus_one)
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, partial_x)
                    res += self.mutate_expr(x)
                    logx = loma_ir.Call('log', [x])
                    partial_y = loma_ir.BinaryOp(loma_ir.Mul(), node, logx)
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), tmp_adjoint, partial_y)
                    res += self.mutate_expr(y)

                    self.adjoint = tmp_adjoint
                case "int2float":
                    tmp_adjoint = self.adjoint
                    self.adjoint = loma_ir.ConstFloat(0.0)
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint
                case "float2int":
                    tmp_adjoint = self.adjoint
                    self.adjoint = loma_ir.ConstFloat(0.0)
                    res += [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = tmp_adjoint

                case _:
                    # Not terminal functions
                    # HW3:
                    # Get arg types IN or OUT
                    primary_func_def = funcs[func_id]
                    assert isinstance(primary_func_def, loma_ir.FunctionDef)
                    diff_args = []
                    for i in range(len(node.args)):
                        arg = node.args[i]
                        arg_def = primary_func_def.args[i]
                        if arg_def.i == loma_ir.In():
                            diff_args.append(arg)
                            diff_args.append(get_lhs_adjoint_var(arg, self.adjoint_id_dict))
                        else:
                            raise NotImplementedError
                    # add d_return
                    diff_args.append(self.adjoint)
                    # Call the differential function
                    diff_func_id = f"_d_rev_{func_id}"
                    diff_call = loma_ir.Call(diff_func_id, diff_args)
                    # Here, we must return a call stmt instead of a call expression, why?
                    res.append(loma_ir.CallStmt(call=diff_call))

            return res


    return RevDiffMutator().mutate_function_def(func)
