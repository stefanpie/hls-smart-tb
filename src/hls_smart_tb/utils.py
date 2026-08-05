def unwrap[T_unwrap](value: T_unwrap | None) -> T_unwrap:
    if value is None:
        raise ValueError("Value is None")
    return value

