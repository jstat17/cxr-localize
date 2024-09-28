import hashlib
from natsort import natsorted
from pathlib import Path

MD5_FILE_HASHES = [
    ("fe8ed0a6961412fddcbb3603c11b3698", "images_001.tar.gz"),
    ("ab07a2d7cbe6f65ddd97b4ed7bde10bf", "images_002.tar.gz"),
    ("2301d03bde4c246388bad3876965d574", "images_003.tar.gz"),
    ("9f1b7f5aae01b13f4bc8e2c44a4b8ef6", "images_004.tar.gz"),
    ("1861f3cd0ef7734df8104f2b0309023b", "images_005.tar.gz"),
    ("456b53a8b351afd92a35bc41444c58c8", "images_006.tar.gz"),
    ("1075121ea20a137b87f290d6a4a5965e", "images_007.tar.gz"),
    ("b61f34cec3aa69f295fbb593cbd9d443", "images_008.tar.gz"),
    ("442a3caa61ae9b64e61c561294d1e183", "images_009.tar.gz"),
    ("09ec81c4c31e32858ad8cf965c494b74", "images_010.tar.gz"),
    ("499aefc67207a5a97692424cf5dbeed5", "images_011.tar.gz"),
    ("dc9fda1757c2de0032b63347a7d2895c", "images_012.tar.gz"),
]

def verify_md5_hashes(chestxray14_gzip_path: Path) -> None:
    """Verify the MD5 hashes of the Chest X-ray 14 gzip files

    Args:
        chestxray14_gzip_path (Path): Path to Chest X-ray 14 gzips
    """
    # sort filenames using natsorted
    md5_info = natsorted(MD5_FILE_HASHES, key=lambda x: x[1])

    for expected_md5, filename in md5_info:
        filepath = chestxray14_gzip_path / filename
        
        # calculate the MD5 hash of the file
        md5 = hashlib.md5()

        print(filepath, end="", flush=True)
        try:
            with open(filepath, 'rb') as file:
                # read the file in chunks to avoid memory issues with large files
                for chunk in iter(lambda: file.read(4096), b''):
                    md5.update(chunk)
            
            # get the calculated MD5 hash
            calculated_md5 = md5.hexdigest()
            
            # compare the calculated MD5 hash with the expected one
            if calculated_md5 == expected_md5:
                print(" ..okay")
            else:
                print(f": DIFFERENT MD5 HASH\n[Expected: {expected_md5}, Calculated: {calculated_md5}]")

        except FileNotFoundError:
            print(": File not found")

        except Exception as e:
            print(f": Error occurred - {e}")
