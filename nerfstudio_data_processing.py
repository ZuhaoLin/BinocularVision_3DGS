from nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset
import pathlib

def main():
    data_dir = pathlib.Path(r'./Data/Videos/IMG_5463.MOV')
    output_dir = pathlib.Path(r'./Data/Training_Data/IMG_5463/')

    processor = VideoToNerfstudioDataset(data_dir, output_dir)
    processor.num_frames_target = 600
    processor.main()

    

if __name__ == "__main__":
    main()