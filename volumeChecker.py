import os

def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Skip broken symlinks
                total_size += os.path.getsize(fp)
    return total_size

def check_virtual_volume(rootdir, virtpath, limit_mb=500):
    """
    Check if the total size of the virtual data folder stays within limit_mb megabytes.
    
    Parameters:
      rootdir (str): The base root directory.
      virtpath (str): The virtual data folder path relative to rootdir.
      limit_mb (int): The size limit in MB (default is 500 MB).
    """
    virt_directory = rootdir + virtpath
    total_virtual_size = get_folder_size(virt_directory)
    limit = limit_mb * 1024 * 1024  # convert MB to bytes

    if total_virtual_size <= limit:
        # print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、{limit_mb}MB以下です。")
        return True
    else:
        print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、{limit_mb}MBを超えています。")
        return False
if __name__ == '__main__':
    rootdir = r'/root/Virtual_Data_Generation'
    virtpath = r'/data/virtual'
    check_virtual_volume(rootdir, virtpath)
