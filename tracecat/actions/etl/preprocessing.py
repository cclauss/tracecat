from collections import deque


def flatten_json(
    data: dict | list[dict], sep="."
) -> dict[str, str | int | float | bool]:
    stack = deque([("", data)])
    items = {}
    append = stack.append
    pop = stack.pop

    while stack:
        path, current = pop()
        if isinstance(current, dict):
            for k, v in current.items():
                new_key = f"{path}{sep}{k}" if path else k
                append((new_key, v))
        elif isinstance(current, list):
            for i, v in enumerate(current):
                new_key = f"{path}[{i}]"
                append((new_key, v))
        else:
            items[path] = current

    return items
