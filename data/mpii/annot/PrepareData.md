## Instructions to create annotation file

First, split the data into `train`, `val`, and `test` set.

Second, split the `train` into `strong_train` which has ground truth human pose and action and `weak_train` which has ground truth action but predicted human pose.

From the ground truth / predicted annotation mat file for the respective split, convert to hdf5 file with the following tags:

- `imgname`: a string containing image file name.
- `part`: a list containing the ground truth or predicted parts
- `action`: an integer containing action index
- `center`: meta info copied from the mat file
- `scale`: meta info copied from the mat file
 