import pandas as pd, os, random
import warnings

from utils import load_model, get_vector, face_detection
from fcs import FCS
from termcolor import colored

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Benchmark for maximum code fix (Range 1-100 inclusive)
    model = load_model('pre-trained/vggface2.h5')
    range_list = [*range(1,101)]
    data_path = 'data'
    persons_dir = os.listdir(data_path)
    report = []

    for person in persons_dir:
        person_abs_path = os.path.abspath(os.path.join(data_path, person))
        if os.path.isfile(person_abs_path):
            continue

        # Probe persons other than chosen person
        probe_persons_dir = persons_dir.copy()
        probe_persons_dir.remove(person)

        person_pics_dir = os.listdir(person_abs_path)
        random_pic = random.choice(person_pics_dir)
        pic_abs_path = os.path.abspath(os.path.join(person_abs_path, random_pic))
        enrolled_user = get_vector(
            image_path=pic_abs_path,
            classifier=model,
            face_detection=face_detection,
            xml_path='haarcascade_frontalface_default.xml'
            ).flatten()
        
        person_pics_dir.remove(random_pic)
        
        for max_fix in range_list:
            print(f"=== Maximum Number for ECC: {max_fix} ===")
            print("Enrolled User:\n\t", pic_abs_path)
            total_success=0
            total_failed=0
            attempt=0
            fcs = FCS(biometric_template=enrolled_user, maximum_fix=max_fix)
            
            # Loop to the same person pics
            for probe_self in person_pics_dir:
                attempt+=1
                probe_self_abs_path = os.path.abspath(os.path.join(person_abs_path, probe_self))
                if os.path.isfile(person_abs_path):
                    continue

                print(f"{colored('Probe Self', 'green')}:\n\t", probe_self_abs_path, end="\n\t")
                probe_self = get_vector(
                    image_path=probe_self_abs_path,
                    classifier=model,
                    face_detection=face_detection,
                    xml_path='haarcascade_frontalface_default.xml'
                ).flatten()
                res = fcs.verify(probe_self)
                if res: total_success +=1 
                else: total_failed+=1
            
            frr = total_failed/attempt
            total_success=0
            total_failed=0
            attempt=0
            print()

            # Loop to other probe person pics
            for probe_person in probe_persons_dir:
                attempt+=1
                probe_abs_path = os.path.abspath(os.path.join(data_path, probe_person))
                if os.path.isfile(probe_abs_path):
                    continue
                probe_pics = os.listdir(probe_abs_path)
                random_ppic = random.choice(probe_pics)
                if random_ppic == random_pic:
                    probe_pics.remove(random_ppic)
                    random_ppic = random.choice(probe_pics)

                ppic_abs_path = os.path.abspath(os.path.join(probe_abs_path, random_ppic))
                print(f"{colored('Probe User', 'red')}:\n\t", ppic_abs_path, end="\n\t")
                probe_user = get_vector(
                    image_path=ppic_abs_path,
                    classifier=model,
                    face_detection=face_detection,
                    xml_path='haarcascade_frontalface_default.xml'
                    ).flatten()

                res = fcs.verify(probe_b_template=probe_user)
                if res: total_success +=1 
                else: total_failed+=1

            far = total_success/attempt
            report.append((max_fix, frr, far))

            print(f"*** Summary for ECC with Max Fix: {max_fix} ***\n\tFalse Acceptance Rate: {far}\n\tFalse Rejection Rate: {frr}")
            print()
            print()

    pd.DataFrame(report, columns=['Max Fix', 'FRR', 'FAR']).groupby(by=['Max Fix']).mean().to_csv('benchmark_report.csv')