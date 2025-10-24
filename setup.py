# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['histpat_toolkit',
 'histpat_toolkit.image_pyramid',
 'histpat_toolkit.patch_extraction',
 'histpat_toolkit.patch_extraction.anchorizers',
 'histpat_toolkit.patch_extraction.labelers',
 'histpat_toolkit.patch_extraction.modifiers',
 'histpat_toolkit.patch_extraction.rectangularizers',
 'histpat_toolkit.tissue_detection',
 'histpat_toolkit.util']

package_data = \
{'': ['*'], 'histpat_toolkit.tissue_detection': ['nn_models/*']}

install_requires = \
['numpy>=1.25.0',
 'opencv-contrib-python>=4.8.0.74',
 'openvino>=2025.3.0,<2026.0.0',
 'pydantic>=2.4.2',
 'svgwrite>=1.4.3,<2.0.0',
 'tcod>=16.2.1']

setup_kwargs = {
    'name': 'histpat-toolkit',
    'version': '0.5.1',
    'description': 'A set of tools to process and analyze histopathology scans',
    'long_description': '## Set up environment\n\nYou need to have `poetry` installed: `pip install poetry`\n\n```\npoetry install\n```\n\n## Testing the library\n\nYou can create notebooks in `notebooks/` folder. Run them with VS Code (make sure that correct kernel is selected) or with Jupyter Lab by command `poetry run jupyter lab`.',
    'author': 'Jarosław Kwiecień',
    'author_email': 'jaroslaw.kwiecien@cancercenter.ai',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.11',
}
from build_cython import *
build(setup_kwargs)

setup(**setup_kwargs)
