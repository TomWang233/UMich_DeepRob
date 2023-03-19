import os
import zipfile

_P1_FILES = [
    "knn.py",
    "knn.ipynb",
    "linear_classifier.py",
    "linear_classifier.ipynb",
    "svm_best_model.pt",
    "softmax_best_model.pt",
]


def make_p1_submission(assignment_path, uniquename=None, umid=None):
    _make_submission(assignment_path, _P1_FILES, "P1", uniquename, umid)


def _make_submission(
    assignment_path, file_list, assignment_no, uniquename=None, umid=None
):
    if uniquename is None or umid is None:
        uniquename, umid = _get_user_info(uniquename, umid)
    zip_path = "{}_{}_{}.zip".format(uniquename, umid, assignment_no)
    zip_path = os.path.join(assignment_path, zip_path)
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename)


def _get_user_info(uniquename, umid):
    if uniquename is None:
        uniquename = input("Enter your uniquename (e.g. topipari): ")
    if umid is None:
        umid = input("Enter your umid (e.g. 12345678): ")
    return uniquename, umid
