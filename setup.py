import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="pavo",
    version="0.0.1",
    author="Son Nguyen",
    author_email="sonnhfit@gmail.com",
    description="Package to create pavo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'ffmpeg-python==0.2.0',
        'requests==2.32.3',
        'retrying==1.3.4',
        'tqdm==4.66.4',
        'botocore==1.29.109',
        'boto3==1.26.109',
        'pillow==10.4.0'
    ]
)
