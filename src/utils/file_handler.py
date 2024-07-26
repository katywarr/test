import os


def check_output_dir(output_dir: str):
    # Exit immediately if the output path does not exist.
    if not os.path.exists(output_dir):
        print('Error: Output directory for does not exist: {}\nCurrent working directory is: {}'
              .format(output_dir, os.getcwd()))
        print('       Check the working directory is set correctly for the relative path and that the dir exists.')
        raise AttributeError('The output directory does not exist.')


def create_dir(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except FileNotFoundError:
            print('Error: Unable to create the directory {} because '
                  'its parent directory does not exist.\n'
                  '(The current working directory is: {})'.format(dir, os.getcwd()))
            raise FileNotFoundError