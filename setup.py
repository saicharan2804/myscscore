from setuptools import setup, find_packages

setup(
    name='myscscore',  # Replace with your package name
    version='0.1.0',  # The current version of your package
    author='SaiC',  # Replace with your name
    author_email='saicharanhahaha@gmail.com',  # Replace with your email address
    description='A package for scoring molecular genetics evaluations',  # Provide a short description
    url='https://github.com/saicharan2804/myscscore',  # Replace with the URL to your repository
    packages=find_packages(),  # Finds all packages in the directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum version of Python required
    install_requires=[
        'numpy',  # Specify your dependencies here
        'rdkit',  # You will need to make sure that rdkit can be installed via pip or adjust accordingly
    ],
    include_package_data=True,  # This will include non-code files specified in MANIFEST.in
    package_data={
        # If there are data files included in packages that need to be installed
        'myscscore': ['model.ckpt-10654.as_numpy.json.gz'],
    },
)

