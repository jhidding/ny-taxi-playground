NY Taxi data playground
=======================

This playground uses [UV](https://docs.astral.sh/uv/) to manage Python packages and [Entangled](https://entangled.github.io) to do literate programming.

Entangled starter
-----------------

Run

```bash
uv run entangled watch
```

Create a new markdown file in the `docs` directory. You can annotate code blocks to create Python files:

~~~markdown
For example, this is "Hello, World" in Python:

```python
#| file: src/hello.py
print("Hello, World!")
```
~~~

You can also give snippets a name using a `#| id: name` tag at the top of a code block, and then `<<name>>` inside another code block to insert its contents there.

Zensical
--------

The contents of `docs` is rendered into HTML pages using the `zensical` engine. To preview locally:

```bash
uv run zensical serve
```

License
-------

Apache 2.0
