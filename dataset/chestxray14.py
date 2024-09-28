import hashlib
from natsort import natsorted
from pathlib import Path

## Global constants
# list of tuples from md5 hash to gzip
MD5_FILE_HASHES = [
    ("fe8ed0a6961412fddcbb3603c11b3698", "vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"), # images_001.tar.gz
    ("ab07a2d7cbe6f65ddd97b4ed7bde10bf", "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz"), # images_002.tar.gz
    ("2301d03bde4c246388bad3876965d574", "f1t00wrtdk94satdfb9olcolqx20z2jp.gz"), # images_003.tar.gz
    ("9f1b7f5aae01b13f4bc8e2c44a4b8ef6", "0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz"), # images_004.tar.gz
    ("1861f3cd0ef7734df8104f2b0309023b", "v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz"), # images_005.tar.gz
    ("456b53a8b351afd92a35bc41444c58c8", "asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz"), # images_006.tar.gz
    ("1075121ea20a137b87f290d6a4a5965e", "jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz"), # images_007.tar.gz
    ("b61f34cec3aa69f295fbb593cbd9d443", "tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz"), # images_008.tar.gz
    ("442a3caa61ae9b64e61c561294d1e183", "upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz"), # images_009.tar.gz
    ("09ec81c4c31e32858ad8cf965c494b74", "l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz"), # images_010.tar.gz
    ("499aefc67207a5a97692424cf5dbeed5", "hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz"), # images_011.tar.gz
    ("dc9fda1757c2de0032b63347a7d2895c", "ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"), # images_012.tar.gz
]


## Data management functions
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
