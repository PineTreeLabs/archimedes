import jinja2
import os
import re
import abc


DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "_templates",
)


def _extract_protected_regions(file_path):
    """
    Extract protected regions from an existing file.
    
    Args:
        file_path: Path to the file with protected regions
        
    Returns:
        Dictionary mapping region names to their content
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    protected_regions = {}
    pattern = r'// PROTECTED-REGION-START: (\w+)(.*?)// PROTECTED-REGION-END'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        region_name = match.group(1)
        region_content = match.group(2).strip()
        protected_regions[region_name] = region_content
    
    return protected_regions


class RendererBase(metaclass=abc.ABCMeta):
    def __init__(self, template_path=None):
        if template_path is None:
            template_path = os.path.join(DEFAULT_TEMPLATE_PATH, self.default_template_name)
        self.template_path = template_path

    @property
    @abc.abstractmethod
    def default_template_name(self):
        """Default template name for this renderer."""

    @property
    @abc.abstractmethod
    def default_output_path(self):
        """Default output file name."""

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
            output_path = self.default_output_path

        context["driver_name"] = os.path.basename(output_path)

        # Extract existing protected regions if the file exists
        protected_regions = {}
        if os.path.exists(output_path):
            protected_regions = _extract_protected_regions(output_path)

        # Add protected regions to the context
        context['protected_regions'] = protected_regions

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


class CDriverRenderer(RendererBase):
    @property
    def default_template_name(self):
        return "c_driver.j2"

    @property
    def default_output_path(self):
        return "main.c"


class ArduinoRenderer(RendererBase):
    @property
    def default_template_name(self):
        return "arduino.j2"

    @property
    def default_output_path(self):
        return "sketch.ino"


_builtin_templates = {
    "c": CDriverRenderer,
    "arduino": ArduinoRenderer,
}

def _render_template(
    driver: str | RendererBase,
    context: dict,
    template_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """
    Render a template with the given context and save it to the specified path.

    Args:
        driver: Name of the driver template to render
        context: Dictionary with template variables
        output_path: Path where the generated code will be written
        template_path: Path to the Jinja2 template file
    """
    if isinstance(driver, str):
        if driver not in _builtin_templates:
            raise ValueError(f"Template '{driver}' not found.")

        renderer = _builtin_templates[driver](template_path)

    else:
        try:
            # Will also raise a TypeError if driver is not a class
            if issubclass(driver, RendererBase):
                renderer = driver(template_path)
            else:
                raise TypeError
        except TypeError:
            raise ValueError("Driver must be a string or RendererBase class.")

    renderer(context, output_path)

    return renderer