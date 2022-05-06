# Robotics: Planning & Navigation: Assignment 1 

This assignment is designed to get you familiar with RRT, a widely-used technique in robotic path planning. You are required to attempt both questions mentioned in the [Assignment PDF](/Assignment1.pdf)

## Instructions:

* Submit your code files and a report in the GitHub classroom repository. Do not upload the simulation videos to the assignment repository.

* Provide the OneDrive/Google Drive links to the generated simulation videos in the report or in the repository README.

* For stitching simulation images into videos:

In `mkmovie.sh` enter the path to the directory where the snapshots are stored and edit the name of the simulation. For example, if my simulation screenshots are stored in the directory `RRT_tree/`, I will run the following command:

`ffmpeg -r 2 -f image2 -i RRT_tree/snap%d.png -s 1000x1000 -pix_fmt yuv420p -y simulation.mp4`


### Deadline: 

**February 18, 2022, 11:55pm**
