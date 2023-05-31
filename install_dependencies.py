import subprocess

def install_package(package):
    subprocess.call(['pip', 'install', package])

# Install numpy
install_package('numpy')
