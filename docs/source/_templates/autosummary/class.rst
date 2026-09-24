{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{# Methods and attributes are documented inline on the class page (listed
    explicitly so autodoc picks the right documenter for each and includes
    inherited members); the summary tables link to those inline entries.
    Nested classes still get their own pages. #}
.. autoclass:: {{ module }}::{{ objname }}
   :members: __init__
   {%- for item in methods if item != '__init__' %}, {{ item }}{% endfor %}
   {%- for item in attributes %}, {{ item }}{% endfor %}
   :member-order: groupwise

   {% block classes %}
   {% set nested_classes = nested_classes.get(module ~ "." ~ objname, []) %}
   {% if nested_classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
   {% for item in nested_classes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods | reject('equalto', '__init__') | list %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {%- if item != '__init__' %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
