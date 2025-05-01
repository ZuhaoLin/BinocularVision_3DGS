from ultralytics import YOLO
import glob
import cv2 as cv

def main():
    leye_paths = glob.glob('images/detection_tests/leye*fove*')
    reye_paths = glob.glob('images/detection_tests/reye*fove*')
    combined_paths = glob.glob('images/detection_tests/combined*fove*')

    print(leye_paths)

    leye_paths.sort()
    reye_paths.sort()
    combined_paths.sort()

    model = YOLO('yolo11n.pt')

    for leye_path, reye_path, combined_path in zip(leye_paths, reye_paths, combined_paths):
        results = model([leye_path, reye_path, combined_path])

        for result in results:
            result.show()
            
        input('Press key to continue')
    

if __name__ == "__main__":
    main()