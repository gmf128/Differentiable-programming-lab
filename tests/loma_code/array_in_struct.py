class Foo:
    arr : Array[int]

def array_in_struct(f : Foo) -> int:
    return f.arr[0] + f.arr[1]
