import sys
import ast
import argparse
import importlib
import inspect
from textwrap import dedent
from typing import Any
from contextlib import contextmanager
from string import whitespace
import re

import logging


logger = logging.getLogger(__name__)


class MyVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        super().__init__()
        self.source = source
        # Line number counting in ast starts at 1
        self.source_lines = [''] + source.splitlines(keepends=False)
        self.stack = []
        self.section_chars = '#*=-^"' + '"' * 1000
        self.flair_pattern = r'^\|[a-zA-Z_-]+\|$'
        self.all_filter = []

    @property
    def depth(self):
        return len(self.stack)

    @contextmanager
    def stacker(self, node):
        self.stack.append(node)
        try:
            yield
        finally:
            self.stack.pop()

    def make_titletext(self, text, prefix='', suffix='') -> str:
        text = prefix + text + suffix
        underline_len = len(text.encode())
        return (
            '\n' + text + '\n' + self.section_chars[self.depth] * underline_len
        )

    def dig(self, node):
        with self.stacker(node):
            super().generic_visit(node)

    def text_from_node_to_before_body(self, node) -> str:
        # text = ' '.join(self.source_lines[node.lineno:node.body[0].lineno])
        lines = self.source_lines[node.lineno:node.body[0].lineno]
        lines = map(str.strip, lines)
        lines = [re.sub(r',$', ', ', l) for l in lines]
        text = ''.join(lines)
        return text.strip(whitespace + ':')

    def generic_visit(self, node: ast.AST) -> Any:
        logger.info(' ' * 4 * len(self.stack) + 'visited %s', node)
        self.dig(node)

    def visit_Module(self, node: ast.Module) -> Any:
        logger.info('visited module', node)
        self.dig(node)

    def stack_contains_a_function_before_a_class(self) -> False:
        for node in reversed(self.stack):
            if isinstance(node, ast.ClassDef):
                return False
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return True
        else:
            return False

    def scoop_flair_items_from_first_body_expr(self, node):
        # Extract flair tags out of docstring. Note: we don't print out the
        # docstring here, that will be left to the ``visit_Expr`` method.
        # We just want the flair so that the title can be modified
        docstring = ast.get_docstring(node)
        flair_items = []
        if docstring:
            flair_items = re.findall(self.flair_pattern, docstring, flags=re.MULTILINE)

        flair_text = ' '.join(flair_items) + (' ' if flair_items else '')
        return flair_text

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.stack_contains_a_function_before_a_class():
            # No documentation for nested functions
            return

        if node.name.startswith('_'):
            return

        if self.depth == 1 and node.name not in self.all_filter:
            return

        text = self.text_from_node_to_before_body(node)
        text = re.sub(r'^def ', '', text)

        anchor_text = node.name
        if isinstance(self.stack[-1], ast.ClassDef):
            class_name = self.stack[-1].name
            text = f'{class_name}.{text}'
            anchor_text = f'{class_name}.{anchor_text}'

        decs = ''
        if node.decorator_list:
            logger.info(node.decorator_list)
            decs = ' '.join(
                '@' + n.name if hasattr(n, 'name') else '@' + n.id
                for n in node.decorator_list
            ) + ' '
        text = decs + text

        flair_text = self.scoop_flair_items_from_first_body_expr(node)
        text = flair_text + '``' + text + '``'

        print()
        print(f'.. _{anchor_text}:')
        print()
        print(self.make_titletext(text))

        # Recurse
        self.dig(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        logger.info('visited async function %s', node)
        self.visit_FunctionDef(node)
        self.dig(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if node.name.startswith('_'):
            return

        if self.depth == 1 and node.name not in self.all_filter:
            return

        anchor_text = node.name
        print(f'.. _{anchor_text}:')
        print()

        text = self.text_from_node_to_before_body(node)
        flair_text = self.scoop_flair_items_from_first_body_expr(node)
        text = flair_text + '``' + text + '``'
        print(self.make_titletext(text))
        self.dig(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Handles docstrings and any other strings that might just appear
        in various places."""
        is_constant_str = isinstance(node.value, ast.Constant) \
            and isinstance(node.value.value, str)
        if not is_constant_str:
            return

        # First line might have different indentation!
        first_line, _, other_lines = node.value.value.partition('\n')
        text = first_line.strip() + '\n' + dedent(other_lines)
        text = re.sub(self.flair_pattern, '', text, flags=re.MULTILINE)
        print(text)

    def visit_Assign(self, node: ast.Assign) -> Any:
        """This is only used to collect the __all__ values to filter
        the output for those."""
        if not hasattr(node.targets[0], 'id'):
            return

        if node.targets[0].id != '__all__':
            return

        self.all_filter = [
            n.value for n in node.value.elts
        ]


def main(args):
    file_contents = open(args.module).read()
    mod = ast.parse(file_contents, filename=args.module)

    visitor = MyVisitor(file_contents)
    visitor.visit(mod)


def main2(args):
    mod = importlib.import_module(args.module)
    sections = [mod.__doc__]

    include_names = None
    if hasattr(mod, '__all__'):
        include_names = mod.__all__

    for name, element, level in parse(mod, level=1, include_names=include_names):
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


def parse(node, level=0, include_names=None):
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

        allow_emit = level!=1 or (
            not include_names or element.__name__ in include_names
        )
        if allow_emit:
            yield name, element, level
            if hasattr(element, '__doc__'):
                print(name, 'hasdoc')
            if inspect.isclass(element):
                yield from parse(element, level+1)


if __name__ == '__main__':
    logging.basicConfig(level='WARNING', stream=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', help='Module name to document')
    parser.add_argument('-a', '--all', help='Only document entries in __all__')
    args = parser.parse_args()
    main(args)
