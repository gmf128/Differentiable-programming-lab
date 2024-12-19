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

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
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
                self.tmp_declare_stmts.append(loma_ir.Declare(\
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(\
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(\
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(\
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

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
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    # Apply the differentiation.
    class PrimalBuildMutator(irmutator.IRMutator):
        def __init__(self):
            super().__init__()
        def mutate_function_def(self, node):
            self.adjoint_id_dict = {}
            primary_code = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            primary_code = irmutator.flatten(primary_code)
            return [primary_code, self.adjoint_id_dict]
        def mutate_declare(self, node):
            # First, we need to generate z as primal code, and declare the adjoint value _dz
            res = []
            res.append(node)  # primal
            id = node.target
            adjoint_id = f"_d{id}_{random_id_generator()}"
            self.adjoint_id_dict[id] = adjoint_id
            res.append(loma_ir.Declare(adjoint_id, t=node.t))
            return res
        def mutate_stmt(self, node):
            match node:
                case loma_ir.Return():
                    return []
                case loma_ir.Declare():
                    return self.mutate_declare(node)
                case loma_ir.Assign():
                    return self.mutate_assign(node)
                case _:
                    assert False, f'Visitor error: unhandled statement {node}'

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

            return super().mutate_assign(node)

    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2:
            # FunctionDef(string id, arg* args, stmt* body, bool is_simd, type? ret_type)

            new_args = []
            adjoint_id_dict = {}
            for arg in node.args:
                new_args.append(arg)
                if arg.i == loma_ir.In():
                    adjoint_id = f"_d{arg.id}_{random_id_generator()}"
                    adjoint_arg = loma_ir.Arg(id=adjoint_id, t=arg.t, i=loma_ir.Out())
                    adjoint_id_dict[arg.id] = adjoint_id
                    new_args.append(adjoint_arg)
            if node.ret_type == loma_ir.Float():
                d_return_id = f"_dreturn_{random_id_generator()}"
                d_return = loma_ir.Arg(id=d_return_id, t=loma_ir.Float(), i=loma_ir.In())
                new_args.append(d_return)
                self.adjoint = loma_ir.Var(d_return_id)
            else:
                raise NotImplementedError("Function ret type which is not float has not been implemented yet")
            self.adjoint_id_dict = adjoint_id_dict
            new_args = tuple(new_args)
            # Primary mode: Declare intermediate variables
            primary_mutator = PrimalBuildMutator()
            primary_code, declare_dict = primary_mutator.mutate_function_def(node)
            for key in declare_dict.keys():
                self.adjoint_id_dict[key] = declare_dict[key]
            # Reverse mode: We have to visit the stmts reversely
            new_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = primary_code + irmutator.flatten(new_body)
            #
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
            self.adjoint = loma_ir.Var(self.adjoint_id_dict[node.target])
            if node.val is None:
                return []
            return self.mutate_expr(node.val)

        def mutate_assign(self, node):
            # HW2: TODO
            return super().mutate_assign(node)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_call_stmt(self, node):
            # HW3: TODO
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

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
                d_var_update = loma_ir.Assign(target=loma_ir.Var(d_var_id), val=loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(d_var_id), self.adjoint))
            return [d_var_update]

        def mutate_array_access(self, node):
            # HW2: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW2: TODO
            return super().mutate_struct_access(node)

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
            # HW2: TODO
            return super().mutate_call(node)

    return RevDiffMutator().mutate_function_def(func)
