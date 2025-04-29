from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn, mortality_prediction_mimic4_fn, length_of_stay_prediction_mimic4_fn, readmission_prediction_mimic4_fn
import pdb


def prepare_drug_task_data():
    mimicvi = MIMIC4Dataset(
        root="/cis/home/xhan56/mimic2.0/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=False,
        refresh_cache=False,
    )

    print("stat")
    mimicvi.stat()
    print("info")
    mimicvi.info()

    mimic4_sample = mimicvi.set_task(mortality_prediction_mimic4_fn)
    print(mimic4_sample[0])
    pdb.set_trace()

    return mimic4_sample


if __name__ == "__main__":
    data = prepare_drug_task_data()
