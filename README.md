# ct-scan-ml

Machine Learning applied to CT scans.

## `scans_to_tfrecords.py` usage examples

To convert several dicom files to a single tfrecord file with the images downscaled of a factor 4 and converted to a float16 data type:

    $ python scans_to_tfrecords.py dicom /home/anderlini/MedPhys/CPTAC-PDA/ ./data/tcia-0.25-float16.tfrecords --downscale=4 --dtype=float16

To convert several nrrd files to a single tfrecord file with the images downscaled of a factor 2 and converted to a float32 data type:

    $ python scans_to_tfrecords.py nrrd "../directory_dati/Pazienti_*/*/[2-9]_*.nrrd" ./data/scans-0.25-float32.tfrecords --downscale=2 --dtype=float32
