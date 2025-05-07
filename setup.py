"""
    Setup file for MI-Agent.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="materials_informatics_agent",
        use_scm_version={"version_scheme": "no-guess-dev"},
        setup_requires=["setuptools_scm"],
        # ---------------------------------------------------
        # Tell setuptools to look in src/ for your packages:
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        # ---------------------------------------------------
        # Your runtime dependencies (tweak version bounds as needed):
        install_requires=[
            "pandas==2.1.4",
            "numpy==1.26.4",
            "langchain-core==0.3.51",
            "langchain-openai==0.3.12",
            "langchain-experimental==0.3.4",
            "trustcall==0.0.38",
            "pydantic==2.11.3",
            "langgraph==0.3.26",
            "langsmith==0.3.27",
            "pycaret==3.3.2",
            "optuna==4.3.0",
            "optuna-integration==4.3.0",
            "markdown==3.8",
            "pdfkit==1.0.0",
            "matplotlib==3.7.5",
            "seaborn==0.13.2",
            "python-dotenv==1.1.0",
        ],
        # ---------------------------------------------------
        # Console‐script entrypoint:
        entry_points={
            "console_scripts": [
                # now running `mi-agent ...` will invoke mi_agent.__main__.main()
                "materials_informatics_agent = mi_agent.__main__:main",
                "mi_agent = mi_agent.__main__:main",
            ],
        },
        # ---------------------------------------------------
        # Metadata (fill in as desired)
        author="Hasan Sayeed",
        description="A LangGraph/LLM‐driven EDA → AutoML → report pipeline",
        long_description=open("README_PYPI.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/hasan-sayeed/MI-Agent",
        classifiers=[
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.10",
    )
