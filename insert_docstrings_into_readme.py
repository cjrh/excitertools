if __name__ == '__main__':
    import excitertools

    output = []

    output.append(
        excitertools.__doc__
    )

    for element in dir(excitertools.Iter):
        print(element)
        d = getattr(excitertools.Iter, element).__doc__
        if d:
            output.append(f'{element}\n{"=" * len(element)}')
            output.append(d)

    for x in output:
        print()
        print(x)
        print()
