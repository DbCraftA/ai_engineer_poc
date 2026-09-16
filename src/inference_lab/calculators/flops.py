


def matmul_output_shape(lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
    
    if lhs[1] != rhs[0]:
        raise ValueError("The k dimension is not the same ! we can not matmul the two matrices")
    
    return (lhs[0],rhs[1])



def matmul_flops(m_nombre_token: int, output_dim: int, k_hidden_dim: int) -> int:
    return 2 * m_nombre_token * k_hidden_dim * output_dim



def matmul_arithmetic_intensity(m: int, n: int, k: int, bytes_per_element: int) -> float:
    flops = matmul_flops(m,n,k)
    octets = ( (m * n) + (n * k) + (m * k) ) * bytes_per_element 
    return flops / octets