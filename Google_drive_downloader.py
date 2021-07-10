from google_drive_downloader import GoogleDriveDownloader as gdd
import os
current_dir = os.path.dirname(os.path.realpath('app.py'))

gdd.download_file_from_google_drive(file_id='1-u5Wd0JJ9JA2YQ0u-G-DcH8G7ExgveYX',
                                    dest_path= os.path.join(current_dir,'sentiment_weight_file.hdf5'),
                                    unzip=False)
gdd.download_file_from_google_drive(file_id='1kvjXNCTzhk-v8Gv8gvLFHiIUnQKkHmDP',
                                    dest_path= os.path.join(current_dir,'Model_3.hdf5'),
                                    unzip=False)
