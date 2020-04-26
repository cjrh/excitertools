import argparse
import importlib
import inspect
from typing import Optional


def main(args):
    mod = importlib.import_module(args.module)
    sections = [mod.__doc__]

    for name, element, level in parse(mod, level=1):
        if name.startswith('__'):
            continue

        if type(element).__name__ == 'module':
            continue

        if type(element).__name__.startswith('TypeVar'):
            continue

        if hasattr(element, '__module__') and element.__module__ != mod.__name__:
            continue

        # print(
        #     name,
        #     type(element).__name__,
        #     type(element), element,
        #     getattr(element, '__module__', '(no module)')
        # )

        if name != 'mro':
            title_text = make_titletext(name, element, level)
            sections.append(title_text)
            section_doc = inspect.getdoc(element)
            if section_doc:
                sections.append(section_doc)

    # print('***')
    print('\n'.join(sections))


def make_titletext(name, element, level):
    section_chars = '#*=-^"'

    text = name

    if inspect.isclass(element):
        text = f'class {name}'
    elif inspect.ismethod(element):
        # class methods
        sig = inspect.signature(element)
        text = f'*classmethod* ``{element.__qualname__}{sig}``'
    elif inspect.isfunction(element):
        # class methods
        sig = inspect.signature(element)
        text = f'``{element.__qualname__}{sig}``'
    else:
        sig = inspect.signature(element)
        text = f'``{element.__qualname__}{sig}``'

    return '\n' + text + '\n' + section_chars[level] * len(text)


def parse(node, level=0):
    def elems_by_appearance(node):
        if node.__name__ == 'type':
            return

        def line_number(node):
            try:
                return inspect.getsourcelines(node)[1]
            except TypeError:
                return 0

        def subnodes(node):
            for n in dir(node):
                try:
                    yield getattr(node, n)
                except AttributeError:
                    continue

        yield from sorted(subnodes(node), key=line_number)

    for element in elems_by_appearance(node):
        try:
            name = element.__name__
        except AttributeError:
            continue

        if name.startswith('_'):
            continue

        yield name, element, level

        if inspect.isclass(element):
            yield from parse(element, level+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', help='Module name to document')
    args = parser.parse_args()
    main(args)
