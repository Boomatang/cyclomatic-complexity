import math


def list_to_columns(data, cols=3):
    if len(data) < cols:
        cols = len(data)
    maxis = []
    col_length = math.ceil(len(data) / cols)
    col_data = []
    for i in range(cols):
        t = data[i * col_length:(i + 1) * col_length]
        col_data.append(t)
        a = len(t[0])
        for item in t:
            if len(item) > a:
                a = len(item)
        maxis.append(a)

    out = ''
    for i in range(col_length):
        line = ''
        for j in range(cols):
            if j + 1 == cols and i >= len(col_data[j]):
                continue
            s = col_data[j][i]
            padding = ' ' * (maxis[j] - len(s))
            line += f"{s}{padding}\t\t"
        out += f"{line}"
        if i != col_length - 1:
            out += '\n'
    return out


if __name__ == '__main__':
    test_data = ['a', 'baass', 'cddd', 'd', 'e','a', 'baass', 'cddd', 'd', 'e','a', 'baass', 'cddd', 'd', 'e','a', 'baass', 'cddd', 'd', 'e','a', 'baass', 'cddd', 'd', 'e']
    print(list_to_columns(test_data))
    test_data = ["a"]
    print(list_to_columns(test_data))
