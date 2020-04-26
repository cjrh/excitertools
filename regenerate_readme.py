import argparse
import importlib
import inspect
from typing import Optional


def main(args):
    mod = importlib.import_module(args.module)
    sections = [mod.__doc__]

    for name, element, level in parse(mod, level=1):
        # print(level, name)
        # if hasattr(mod, '__all__') and name not in mod.__all__:
        #     continue

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

        if name == 'mro':
            continue

        section_doc = inspect.getdoc(element)
        tag_prefix = ''
        if section_doc:
            import re
            pat = r'^\|[a-zA-Z_-]+\|$'
            tags = re.findall(pat, section_doc, flags=re.MULTILINE)
            section_doc = re.sub(pat, '', section_doc, flags=re.MULTILINE)
            if tags:
                tag_prefix = ' '.join(t for t in tags) + '    '

        title_text = make_titletext(name, element, level, prefix=tag_prefix)
        sections.append(title_text)
        # print(name, section_doc)
        if section_doc:
            sections.append(section_doc)

        else:
            sections.append('Docstring TBD')

    # print('***')
    print('\n'.join(sections))


def make_titletext(name, element, level, prefix=''):
    section_chars = '#*=-^"'

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
        try:
            sig = inspect.signature(element)
            text = f'``{element.__qualname__}{sig}``'
        except (ValueError, TypeError):
            text = name

    if prefix:
        text = prefix + text

    underline_len = len(text.encode())

    return '\n' + text + '\n' + section_chars[level] * underline_len


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
    parser.add_argument('-a', '--all', help='Only document entries in __all__')
    args = parser.parse_args()
    main(args)
