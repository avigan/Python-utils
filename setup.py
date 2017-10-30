from setuptools import setup

setup(
    name='Python-utils',
    version='0.1',

    description='Miscellaneous Python utilities for image processing, analysis, simulation, etc',
    url='https://github.com/avigan/Python-utils',
    author='Arthur Vigan',
    author_email='arthur.vigan@lam.fr',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='image processing data analysis',
    packages=['vigan', 'vigan.optics', 'vigan.utils','vigan.astro'],
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib'
    ],
    zip_safe=False
)
