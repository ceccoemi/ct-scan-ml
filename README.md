# ct-scan-ml

Machine Learning applied to CT scans.

## Scans conversion

To convert several dicom files to a series of tfrecord files with the images downscaled of a factor 4 (in the `acer-1` server):

    python scans_to_tfrecords.py dicom "../../anderlini/MedPhys/CPTAC-PDA/*/*/*/" ./data/tcia-0.25 --downsample=4

To convert several nrrd files to a series of tfrecord files with the images downscaled of a factor 2 (in the `acer-1` server):

    python scans_to_tfrecords.py nrrd "../directory_dati/Pazienti_*/*/[2-9]_*.nrrd" ./data/nrrd-0.5 --downsample=2

## Training

To train a model, first tune the hyperparamenters in `config.py`, then run

    python train.py

If you are in a multi-GPU system, to run the training process in a specific GPU run

    python train.py --gpu=<GPU-id>
