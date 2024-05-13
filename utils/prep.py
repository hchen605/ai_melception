import os
import gdown, wget
import zipfile, tarfile
import argparse

def download_file(file, destination):
    _, ext = os.path.splitext(file)
    
    if ext == '.gz':
        mode = 'tar'
        wget.download(file, destination)
    else:
        mode = 'zip'
        gdown.download(id=file, output=destination)

    return mode

def unzip_file(file, mode):
    save_dir = os.path.dirname(file)
    
    if mode == 'tar':
        with tarfile.open(file, 'r') as tar:
            tar.extractall(save_dir)
    else:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(save_dir)


def main(args):
    jobs = {
        "13oIf-706vszak0DdiQpTB09btJx7zTIy": "./init/data.zip",  # init pt-weight
        "10E6F8RbRuSSg9wmYiPv6jyDsaFSSuyte": "./arrange/checkpoint.zip",  # arrange pt-weight
        "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz": os.path.join(args.data_path, "lmd_full.tar.gz")  # Training Data
    }
    
    # Download files from Google Drive
    for file, destination in jobs.items():
        if not os.path.exists(destination):
            ## Check save_dir exist
            if not os.path.exists(os.path.dirname(destination)):
                os.mkdir(os.path.dirname(destination))
            print('Downloading...')
            mode = download_file(file, destination)
            print('Unzipping...')
            unzip_file(destination, mode)
            print('Removing...')
            os.remove(destination)
        else:
            print('File exists \nSkipping...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Data', help='path to save training data')
    args = parser.parse_args()
    main(args)


