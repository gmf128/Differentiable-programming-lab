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
def terminal_func_calls(func_id):
    def Dsin(input):
        input_x = input[0]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        val = loma_ir.Call('sin', [x])
        dval = loma_ir.BinaryOp(loma_ir.Mul(), dx, loma_ir.Call('cos', [x]))
        # return loma_ir.Call('make__dfloat', [val, dval])
        return [val, dval]
    def Dcos(input):
        input_x = input[0]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        val = loma_ir.Call('cos', [x])
        dval = loma_ir.BinaryOp(loma_ir.Mul(), dx, loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), loma_ir.Call('sin', [x])))
        return [val, dval]

    def Dpow(input):
        input_x = input[0]
        input_y = input[1]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        y = loma_ir.StructAccess(input_y, 'val')
        dy = loma_ir.StructAccess(input_y, 'dval')
        val = loma_ir.Call('pow', [x, y])
        partial_x = loma_ir.BinaryOp(loma_ir.Mul(), dx, (loma_ir.BinaryOp(loma_ir.Mul(), y, loma_ir.Call('pow', [x, loma_ir.BinaryOp(loma_ir.Sub(), y, loma_ir.ConstInt(1))]))))
        logx = loma_ir.Call('log', [x])
        partial_y = loma_ir.BinaryOp(loma_ir.Mul(), dy, loma_ir.BinaryOp(loma_ir.Mul(), logx, val))
        dval = loma_ir.BinaryOp(loma_ir.Add(), partial_x, partial_y)
        return [val, dval]

    def Dlog(input):
        input_x = input[0]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        val = loma_ir.Call('log', [x])
        dval = loma_ir.BinaryOp(loma_ir.Div(), dx, x)
        return [val, dval]

    def Dexp(input):
        input_x = input[0]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        val = loma_ir.Call('exp', [x])
        dval = loma_ir.BinaryOp(loma_ir.Mul(), dx, val)
        return [val, dval]

    def Dsqrt(input):
        input_x = input[0]
        x = loma_ir.StructAccess(input_x, 'val')
        dx = loma_ir.StructAccess(input_x, 'dval')
        val = loma_ir.Call('sqrt', [x])
        dval = loma_ir.BinaryOp(loma_ir.Div(), dx, loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.ConstFloat(2.0), val))
        return [val, dval]

    def Dfloat2int(input):
        x = input
        val = loma_ir.Call('float2int',[x], t=loma_ir.Int())
        dval = loma_ir.ConstFloat(0.0)
        return [val, dval]

    def Dint2float(input):
        x = input
        val = loma_ir.Call('int2float',[x], t=loma_ir.Float())
        dval = loma_ir.ConstFloat(0.0)
        return [val, dval]

    dfunc_dict = {}
    dfunc_dict['sin'] = Dsin
    dfunc_dict['cos'] = Dcos
    dfunc_dict['pow'] = Dpow
    dfunc_dict['log'] = Dlog
    dfunc_dict['exp'] = Dexp
    dfunc_dict['sqrt'] = Dsqrt
    dfunc_dict['float2int'] = Dfloat2int
    dfunc_dict['int2float'] = Dint2float

    if func_id not in dfunc_dict.keys():
        return None

    return dfunc_dict[func_id]


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
            new_ret_type = loma_to_diff_type(node.ret_type, diff_structs)
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
            # ret type int:
            # ret type float: assemble the results of expr by building a loma_ir.Call to make__dfloat, with the primal and differential expressions being the two arguments.
            match func.ret_type:
                case loma_ir.Int():
                    val, dval = self.mutate_expr(node.val)
                    return loma_ir.Return(val, lineno=node.lineno)
                case loma_ir.Float():
                    val, dval = self.mutate_expr(node.val)
                    ret = loma_ir.Call("make__dfloat", [val, dval])
                    return loma_ir.Return(ret, lineno=node.lineno)
                case _:
                    # ret type Structures
                    ret = self.mutate_expr(node.val)[0]
                    return loma_ir.Return(ret, lineno=node.lineno)

        def mutate_declare(self, node):
            # HW1:
            # Declare (string target, type t, expr* val)
            target = node.target
            diff_t = loma_to_diff_type(node.t, diff_structs)
            match node.t:
                case loma_ir.Int():
                    val, dval = self.mutate_expr(node.val)
                    return loma_ir.Declare(target, diff_t, val, lineno=node.lineno)
                case loma_ir.Float():
                    if node.val is None:
                        return loma_ir.Declare(target, diff_t, None, lineno=node.lineno)
                    else:
                        val, dval = self.mutate_expr(node.val)
                        ret_val = loma_ir.Call("make__dfloat", [val, dval])
                        return loma_ir.Declare( \
                            target,
                            diff_t,
                            ret_val,
                            lineno=node.lineno)
                case _:
                    # Structures
                    if node.val is None:
                        return loma_ir.Declare(target, diff_t, None, lineno=node.lineno)
                    return loma_ir.Declare( \
                        target,
                        diff_t,
                        val=self.mutate_expr(node.val)[0],
                        lineno=node.lineno)
        def mutate_assign(self, node):
            # HW1:
            # expr target = expr val
            match node.target:
                # FIXME ugly, any better resolution?
                case loma_ir.ArrayAccess():
                    target = loma_ir.ArrayAccess( \
                        self.mutate_expr(node.target.array)[0],
                        self.mutate_expr(node.target.index)[0],
                        lineno=node.lineno,
                        t=loma_to_diff_type(node.target.t, diff_structs))

                case _:
                    target = node.target
            val, dval = self.mutate_expr(node.val)
            if node.target.t == loma_ir.Int():
                return loma_ir.Assign(target, val, lineno=node.lineno)
            elif node.target.t == loma_ir.Float():
                ret_val = loma_ir.Call("make__dfloat", [val, dval])
            else:
                ret_val = val
            return loma_ir.Assign(target, ret_val, lineno=node.lineno)

        def mutate_ifelse(self, node):
            # HW3:
            new_cond = self.mutate_expr(node.cond)[0]
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

        def mutate_while(self, node):
            # HW3:
            new_cond = self.mutate_expr(node.cond)[0]
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)
            return loma_ir.While(
                new_cond,
                node.max_iter,
                new_body,
                lineno=node.lineno
            )

        def mutate_const_float(self, node):
            # HW1:
            # (val, 0)
            return [node, loma_ir.ConstFloat(0.0)]

        def mutate_const_int(self, node):
            # HW1:
            # (val, 0)
            return [node, loma_ir.ConstFloat(0.0)]

        def mutate_var(self, node):
            # HW1:
            # Mutate a Variable, and return the tuple (val, dval), we need to use loma_ir.StructAccess to get the member variable(val & dval)
            match node.t:
                # for integers, return (int, 0)
                case loma_ir.Int():
                    return [node, loma_ir.ConstFloat(0.0)]
                # for arrays, no nothing
                case loma_ir.Array():
                    return [node, None]
                case loma_ir.Float():
                    val = loma_ir.StructAccess(node, 'val', lineno=node.lineno)
                    dval = loma_ir.StructAccess(node, 'dval', lineno=node.lineno)
                    return [val, dval]
                case _:
                    return [node, None]

        def mutate_array_access(self, node):
            # HW1:
            match node.t:
                case loma_ir.Int():
                    return [loma_ir.ArrayAccess(self.mutate_expr(node.array)[0], self.mutate_expr(node.index)[0],
                                               lineno=node.lineno, t=node.t),
                            loma_ir.ConstFloat(0.0)]
                case loma_ir.Float():
                    item = loma_ir.ArrayAccess( \
                        self.mutate_expr(node.array)[0],
                        self.mutate_expr(node.index)[0],
                        lineno=node.lineno,
                        t=loma_to_diff_type(node.t, diff_structs))
                    val = loma_ir.StructAccess(item, 'val', lineno=node.lineno)
                    dval = loma_ir.StructAccess(item, 'dval', lineno=node.lineno)
                    return [val, dval]
                case _:
                    return [node, None]

        def mutate_struct_access(self, node):
            # HW1:
            item = loma_ir.StructAccess( \
                self.mutate_expr(node.struct)[0],
                node.member_id,
                lineno=node.lineno,
                t=loma_to_diff_type(node.t, diff_structs))
            match item.t:
                case loma_ir.Int():
                    return (item, loma_ir.ConstFloat(0.0))
                case loma_ir.Struct():
                    if item.t.id == '_dfloat':
                        val = loma_ir.StructAccess(item, 'val', lineno=node.lineno)
                        dval = loma_ir.StructAccess(item, 'dval', lineno=node.lineno)
                        return [val, dval]
                    else:
                        return [item, None]
                case _:
                    return [item, None]


        def mutate_add(self, node):
            # HW1:
            # (x.val + y.val, x.dval + y.dval)
            xval, xdval = self.mutate_expr(node.left)
            yval, ydval = self.mutate_expr(node.right)
            val =  loma_ir.BinaryOp( \
                loma_ir.Add(),
                xval,
                yval,
                lineno=node.lineno,
                )
            dval = loma_ir.BinaryOp( \
                loma_ir.Add(),
                xdval,
                ydval,
                lineno=node.lineno,
                )
            return [val, dval]

        def mutate_sub(self, node):
            # HW1:
            # (x.val - y.val, x.dval - y.dval)
            xval, xdval = self.mutate_expr(node.left)
            yval, ydval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp( \
                loma_ir.Sub(),
                xval,
                yval,
                lineno=node.lineno,
            )
            dval = loma_ir.BinaryOp( \
                loma_ir.Sub(),
                xdval,
                ydval,
                lineno=node.lineno,
            )
            return [val, dval]

        def mutate_call_stmt(self, node):
            return loma_ir.CallStmt( \
                self.mutate_expr(node.call)[0],
                lineno=node.lineno)

        def mutate_mul(self, node):
            # HW1:
            # (x.val * y.val, x.dval * y.val + x.val * y.dval)
            xval, xdval = self.mutate_expr(node.left)
            yval, ydval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                xval,
                yval,
                lineno=node.lineno,
            )
            xdy = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                xval,
                ydval,
                lineno=node.lineno,
            )
            ydx = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                xdval,
                yval,
                lineno=node.lineno,
            )
            dval = loma_ir.BinaryOp( \
                loma_ir.Add(),
                xdy,
                ydx,
                lineno=node.lineno,
            )
            return [val, dval]

        def mutate_div(self, node):
            # HW1:
            # (x.val / y.val, (ydx - xdy) / y^2)
            xval, xdval = self.mutate_expr(node.left)
            yval, ydval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp( \
                loma_ir.Div(),
                xval,
                yval,
                lineno=node.lineno,
            )
            xdy = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                xval,
                ydval,
                lineno=node.lineno,
            )
            ydx = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                xdval,
                yval,
                lineno=node.lineno,
            )
            y_sqr = loma_ir.BinaryOp( \
                loma_ir.Mul(),
                yval,
                yval,
                lineno=node.lineno,
            )
            dval = loma_ir.BinaryOp( \
                loma_ir.Div(),
                loma_ir.BinaryOp(loma_ir.Sub(), ydx, xdy, lineno=node.lineno),
                y_sqr,
                lineno=node.lineno,
            )
            return [val, dval]

        def mutate_call(self, node):
            # HW1:
            # before: y = f(x)
            # after: (y.val, y.dval) = Df(x.val, x.dval);
            func_id = node.id
            def mutate_arg(arg):
                match arg.t:
                    case loma_ir.Float():
                        val, dval = self.mutate_expr(arg)
                        return loma_ir.Call("make__dfloat", [val, dval])
                    case _:
                        val, _ = self.mutate_expr(arg)
                        return val
            if terminal_func_calls(func_id) is not None:
                # Terminal functions
                diff_func = terminal_func_calls(func_id)
                match func_id:
                    case "int2float":
                        val, dval = self.mutate_expr(node.args[0])
                        return diff_func(val)
                    case "float2int":
                        val, dval = self.mutate_expr(node.args[0])
                        return diff_func(val)
                    case _:
                        diff_args = [mutate_arg(arg) for arg in node.args]
                        return diff_func(diff_args)
            else:
                # Get arg types IN or OUT
                primary_func_def = funcs[func_id]
                assert isinstance(primary_func_def, loma_ir.FunctionDef)
                diff_args = []
                for i in range(len(node.args)):
                    arg = node.args[i]
                    arg_def = primary_func_def.args[i]
                    if arg_def.i == loma_ir.In():
                        diff_args.append(mutate_arg(arg))
                    else:
                        diff_args.append(arg)

                # Call the differential function
                diff_func_id = f"_d_fwd_{func_id}"
                ret_type = primary_func_def.ret_type
                match ret_type:
                    case loma_ir.Float():
                        dfloat_ret = loma_ir.Call(diff_func_id, args=diff_args, lineno=node.lineno)
                        return [loma_ir.StructAccess(struct=dfloat_ret, member_id='val'), loma_ir.StructAccess(struct=dfloat_ret, member_id='dval')]
                    case None:
                        return [loma_ir.Call(diff_func_id, args=diff_args, lineno=node.lineno),]
                    case loma_ir.Struct():
                        return [loma_ir.Call(diff_func_id, args=diff_args, lineno=node.lineno), None]
                    case _:
                        raise NotImplementedError

        def mutate_less(self, node):
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return [loma_ir.BinaryOp(loma_ir.Less(), lval, rval, lineno=node.lineno), ]

        def mutate_less_equal(self, node):
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return [loma_ir.BinaryOp(loma_ir.LessEqual(), lval, rval, lineno=node.lineno), ]

        def mutate_greater(self, node):
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return [loma_ir.BinaryOp(loma_ir.Greater(), lval, rval, lineno=node.lineno), ]

        def mutate_greater_equal(self, node):
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return [loma_ir.BinaryOp(loma_ir.GreaterEqual(), lval, rval, lineno=node.lineno), ]

        def mutate_equal(self, node):
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return [loma_ir.BinaryOp(loma_ir.Equal(), lval, rval, lineno=node.lineno), ]

        def mutate_and(self, node):
            return super().mutate_and(node)

        def mutate_or(self, node):
            return super().mutate_or(node)

    return FwdDiffMutator().mutate_function_def(func)
