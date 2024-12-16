import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def loma_to_diff_type(t : loma_ir.type,
                        diff_structs : dict[str, loma_ir.Struct]) -> loma_ir.Struct:
    """ Given a loma type, maps to the corresponding differential Struct by looking up diff_structs
    """

    match t:
        case loma_ir.Int():
            return diff_structs['int']
        case loma_ir.Float():
            return diff_structs['float']
        case loma_ir.Array():
            return loma_ir.Array(loma_to_diff_type(t.t, diff_structs), t.static_size)
        case loma_ir.Struct():
            return diff_structs[t.id]
        case None:
            return None
        case _:
            assert False

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

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
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1:
            # FunctionDef(string id, arg* args, stmt* body, bool is_simd, type? ret_type)
            # The mutated function:
            #   id: diff_func_id
            #   args: Diff[args]
            #   body: recursively call mutate_stmt()
            #   is_simd: remain unchanged
            #   ret_type: _dfloat
            #   lineno: the lineno of Forward/Backward func

            # Diff[args], see tool function:
            new_args = []
            for arg in node.args:
                diff_arg = loma_ir.Arg(id=arg.id, t=loma_to_diff_type(arg.t, diff_structs), i=arg.i)
                new_args.append(diff_arg)
            new_args = tuple(new_args)
            # Recursively mutate stmt
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = irmutator.flatten(new_body)
            #
            new_ret_type = structs['_dfloat']
            fwd_func_id = func_to_fwd[node.id]
            lineno = funcs[fwd_func_id].lineno

            return loma_ir.FunctionDef( \
                            diff_func_id,
                            new_args,
                            new_body,
                            node.is_simd,
                            new_ret_type,
                            lineno=lineno)

        def mutate_return(self, node):
            # HW1:
            # return expr
            # assemble the results of expr by building a loma_ir.Call to make__dfloat, with the primal and differential expressions being the two arguments.
            val, dval = self.mutate_expr(node.val)
            ret = loma_ir.Call("make__dfloat", [val, dval])
            return loma_ir.Return(ret, lineno = node.lineno)

        def mutate_declare(self, node):
            # HW1: TODO
            return super().mutate_declare(node)

        def mutate_assign(self, node):
            # HW1: TODO
            return super().mutate_assign(node)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            return super().mutate_const_float(node)

        def mutate_const_int(self, node):
            # HW1: TODO
            return super().mutate_const_int(node)

        def mutate_var(self, node):
            # HW1:
            # Mutate a Variable, and return the tuple (val, dval), we need to use loma_ir.StructAccess to get the member variable(val & dval)
            val = loma_ir.StructAccess(node, 'val', lineno=node.lineno)
            dval = loma_ir.StructAccess(node, 'dval', lineno=node.lineno)
            return (val, dval)

        def mutate_array_access(self, node):
            # HW1: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            return super().mutate_struct_access(node)

        def mutate_add(self, node):
            # HW1: TODO
            return super().mutate_add(node)

        def mutate_sub(self, node):
            # HW1: TODO
            return super().mutate_sub(node)

        def mutate_mul(self, node):
            # HW1: TODO
            return super().mutate_mul(node)

        def mutate_div(self, node):
            # HW1: TODO
            return super().mutate_div(node)

        def mutate_call(self, node):
            # HW1: TODO
            return super().mutate_call(node)

    return FwdDiffMutator().mutate_function_def(func)
