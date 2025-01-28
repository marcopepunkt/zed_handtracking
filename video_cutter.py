import pyzed.sl as sl

def main():
    video = sl.Camera()
    video.set_from_svo_file("input/HD1080_SN39725782_10-06-27.svo2")


if __name__ == "__main__":
    main()
    
    