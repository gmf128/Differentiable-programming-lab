def array_output(x : float, y : Array[float]):
    y[0] = x * x
    y[1] = x * x * x

d_array_output = fwd_diff(array_output)
