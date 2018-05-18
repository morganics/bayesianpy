
def exclude(lst, i):
    if i == 0:
        return lst[i+1:]

    return lst[:i] + lst[i+1:]