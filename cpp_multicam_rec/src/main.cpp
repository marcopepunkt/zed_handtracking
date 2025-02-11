///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format.					   **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;

void runner(sl::Camera& zed, bool& exit, int id) {

	RecordingParameters rec_p;
	rec_p.compression_mode = SVO_COMPRESSION_MODE::H264;
	
	auto cam_infos = zed.getCameraInformation();
	auto SN = cam_infos.serial_number;
	std::string svo_name("SVO_SN" + std::to_string(SN) + ".svo");
	// Enable recording with the filename specified in argument
	rec_p.video_filename = String(svo_name.c_str());
	auto err = zed.enableRecording(rec_p);
	if (err != ERROR_CODE::SUCCESS)
		std::cout << "ZED [" << id << "] can not record, " << err << std::endl;

	int fps = cam_infos.camera_configuration.fps;
	std::cout << "ZED[" << id << "]" << " Start recording "<< cam_infos.camera_configuration.resolution.width<<"@"<< fps <<"\n";
	
    int frames_recorded = 0;
	int nb_grabbed = 0;

	auto start = std::chrono::high_resolution_clock::now();
	sl::RecordingStatus state;
    while (!exit) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Each new frame is added to the SVO file
			if (1) {
				state = zed.getRecordingStatus();
				if (state.status) frames_recorded++;
				nb_grabbed++;
				if ((nb_grabbed % fps) == 0)
					std::cout << "Z" << id << ": " << nb_grabbed << "\n";
			}
		}
    }

	zed.disableRecording();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	int nb_frame_theoric = (duration / 1000.f) * fps;

	std::cout << "Recording status ["<< id<<"]: theoric " << nb_frame_theoric << ", grabed " << nb_grabbed << ", saved " << frames_recorded << " frames, avg compression time "<< state.average_compression_time<<"ms\n";
}

int main(int argc, char **argv) {
    // Set configuration parameters for the ZED
    InitParameters initParameters;	
	initParameters.camera_resolution = RESOLUTION::HD1080;
	initParameters.camera_fps = 30;
	initParameters.depth_mode = DEPTH_MODE::NEURAL;
	initParameters.coordinate_units = UNIT::MILLIMETER;
	initParameters.sdk_verbose = 1;

	const int NB_ZED = Camera::getDeviceList().size();

    // Create a ZED camera
    std::vector<Camera> zeds(NB_ZED);
    std::vector<std::thread> pool(NB_ZED);
    std::cout << "Try to open " << NB_ZED << " ZEDs\n";

    for (int z = 0; z < NB_ZED; z++) {
		initParameters.input.setFromCameraID(z);

        // Open the camera
        ERROR_CODE err = zeds[z].open(initParameters);
        if (err != ERROR_CODE::SUCCESS)
            std::cout <<"ZED ["<<z<<"] can not be opened, "<< err<< std::endl;		
		// After opening each camera:
		zeds[z].setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0); // disable auto
		zeds[z].setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, 50); // pick an exposure in [0..100]
		zeds[z].setCameraSettings(sl::VIDEO_SETTINGS::GAIN, 10); // pick a gain in [0..100]

		// Optional: lock white balance as well
		zeds[z].setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO, 0);
		zeds[z].setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE, 5000); // pick a temperature in [2800..6500]
    }

    // Start recording SVO, stop with Ctrl-C command
    std::cout << "SVO is Recording, use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();

	for (int z = 0; z < NB_ZED; z++) 
		if(zeds[z].isOpened())
			pool[z] = std::thread(runner, std::ref(zeds[z]), std::ref(exit_app), z);

    while (!exit_app)
        sl::sleep_us(50);
   
    for (int z = 0; z < NB_ZED; z++)
        if (zeds[z].isOpened())
			if(pool[z].joinable())
				pool[z].join();

	for (int z = 0; z < NB_ZED; z++)
		zeds[z].close();

    return 0;
}
