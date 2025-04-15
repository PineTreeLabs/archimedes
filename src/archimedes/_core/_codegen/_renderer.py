import jinja2
import os


DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "_templates",
)


class CDriverRenderer:
    def __init__(self, template_path=None):
        if template_path is None:
            template_path = os.path.join(DEFAULT_TEMPLATE_PATH, "c_driver.j2")
        self.template_path = template_path

    def __call__(self, context, output_path=None):
        """
        Render a C driver file from a Jinja2 template.
        
        Args:
            context: Dictionary with template variables
            output_path: Path where the generated code will be written
        """
        template_dir = os.path.dirname(self.template_path)
        template_name = os.path.basename(self.template_path)

        if output_path is None:
            output_path = "main.c"

        # Set up Jinja environment
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir or '.'),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Load template
        template = env.get_template(template_name)
        
        # Render template with context
        rendered_code = template.render(**context)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write output file
        with open(output_path, 'w') as f:
            f.write(rendered_code)
        
        print(f"Generated C driver at: {output_path}")



_builtin_templates = {
    "c": CDriverRenderer,
}

def _render_template(
    template_name: str,
    context: dict,
    template_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """
    Render a template with the given context and save it to the specified path.

    Args:
        template_name: Name of the template to render
        context: Dictionary with template variables
        output_path: Path where the generated code will be written
        template_path: Path to the Jinja2 template file
    """
    if template_name not in _builtin_templates:
        raise ValueError(f"Template '{template_name}' not found.")

    renderer = _builtin_templates[template_name](template_path)
    renderer(context, output_path)