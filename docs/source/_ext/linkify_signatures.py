"""Hyperlink the class-name prefix in method/attribute/property signatures."""

from __future__ import annotations

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx


def _linkify(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    py_domain = app.env.get_domain("py")

    for signode in doctree.findall(addnodes.desc_signature):
        classname = signode.get("class")
        if not classname:
            continue
        module = signode.get("module")
        fullname = f"{module}.{classname}" if module else classname

        entry = py_domain.objects.get(fullname)
        if entry is None:
            continue

        for addname in signode.findall(addnodes.desc_addname):
            text = addname.astext()
            if text.rstrip(".") != classname:
                continue

            refuri = app.builder.get_relative_uri(docname, entry.docname)
            refnode = nodes.reference(
                "", "", internal=True, refuri=f"{refuri}#{entry.node_id}"
            )
            refnode += nodes.Text(text)
            addname.clear()
            addname += refnode
            break


def setup(app: Sphinx) -> dict:
    app.connect("doctree-resolved", _linkify)
    return {"version": "1.0", "parallel_read_safe": True}
