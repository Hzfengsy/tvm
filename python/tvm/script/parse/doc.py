import ast
import inspect
import typing
from collections import defaultdict

from . import doc_core as doc
from .doc_core import *  # pylint: disable=unused-import,wildcard-import,redefined-builtin,W0614

FnToDoc = typing.Callable[[ast.AST], doc.AST]
FnFromDoc = typing.Callable[[doc.AST], ast.AST]


class Entry:
    to_doc: typing.Optional[FnToDoc]
    from_doc: typing.Optional[FnFromDoc]

    def __init__(self):
        self.to_doc = None
        self.from_doc = None


class Registry:
    _inst: typing.Optional["Registry"] = None
    table: typing.Dict[str, Entry]

    def __init__(self):
        self.table = defaultdict(Entry)


def register_to_doc(name: str):
    def f(to_doc: FnToDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].to_doc = to_doc

    return f


def register_from_doc(name: str):
    def f(to_doc: FnFromDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].from_doc = to_doc

    return f


def _is_atomic_type(node):
    return (
        node is None
        or node in [..., True, False]
        or isinstance(
            node,
            (
                int,
                float,
                str,
                bool,
                bytes,
                complex,
            ),
        )
    )


def _get_registry_entry(cls_name, attr):
    cls_name = cls_name.split(".")[-1]
    reg = Registry._inst  # pylint: disable=protected-access
    if cls_name in reg.table:
        entry = reg.table[cls_name]
        return getattr(entry, attr, None)
    return None


def from_doc(node):
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(from_doc(n) for n in node)
    if isinstance(node, list):
        return [from_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "from_doc")
    if not func:
        raise NotImplementedError(f"from_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def to_doc(node):
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(to_doc(n) for n in node)
    if isinstance(node, list):
        return [to_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "to_doc")
    if not func:
        raise NotImplementedError(f"to_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def _register_default():
    class DefaultTranslator:
        def __init__(self, doc_cls, func, fields):
            self.doc_cls = doc_cls  # getattr(doc, name)
            self.func = func
            self.fields = fields

        def __call__(self, node):
            kv = {attr: self.func(getattr(node, attr, None)) for attr in self.fields}
            return self.doc_cls(**kv)

    Registry._inst = Registry()  # pylint: disable=protected-access
    for cls_name in dir(doc):
        doc_cls = getattr(doc, cls_name)
        if inspect.isclass(doc_cls) and issubclass(doc_cls, doc.AST):
            assert "." not in cls_name
            register_to_doc(cls_name)(
                DefaultTranslator(
                    getattr(doc, cls_name),
                    to_doc,
                    doc_cls._FIELDS,  # pylint: disable=protected-access
                )
            )
            register_from_doc(cls_name)(
                DefaultTranslator(
                    getattr(ast, cls_name),
                    from_doc,
                    doc_cls._FIELDS,  # pylint: disable=protected-access
                )
            )


def parse(
    source,
    filename="<unknown>",
    mode="exec",
) -> doc.AST:
    try:
        program = ast.parse(
            source=source,
            filename=filename,
            mode=mode,
            feature_version=(3, 8),
        )
    except:
        program = ast.parse(
            source=source,
            filename=filename,
            mode=mode,
        )
    return to_doc(program)


class NodeVisitor:
    def visit(self, node: doc.AST) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        if not isinstance(node, doc.AST):
            return
        return getattr(
            self,
            "visit_" + node.__class__.__name__.split(".")[-1],
            self.generic_visit,
        )(node)

    def generic_visit(self, node: doc.AST) -> None:
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                self.visit(value)


class NodeTransformer:
    def visit(self, node: doc.AST) -> doc.AST:
        if isinstance(node, list):
            return [self.visit(item) for item in node]
        if isinstance(node, tuple):
            return tuple(self.visit(item) for item in node)
        if not isinstance(node, doc.AST):
            return node
        return getattr(
            self,
            "visit_" + node.__class__.__name__.split(".")[-1],
            self.generic_visit,
        )(node)

    def generic_visit(self, node: doc.AST) -> doc.AST:
        kv: typing.Dict[str, typing.Any] = {}
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                value = self.visit(value)
            kv[field] = value
        return node.__class__(**kv)


_register_default()
