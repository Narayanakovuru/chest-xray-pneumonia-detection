from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="pneumonia_detection",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
    )
