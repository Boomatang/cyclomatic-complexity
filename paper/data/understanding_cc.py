def hello_world(name=None):
    message = "hello world"

    if name:
        message = f"hello {name}"

    print(message)


def hello_again(name=None):
    if name:
        print(f"hello {name}")
    else:
        print("hello world")
