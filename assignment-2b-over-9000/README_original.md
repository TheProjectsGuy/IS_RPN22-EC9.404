# Robotics: Planning & Navigation: Assignment 2B

This half of the assignment aims at getting you familiar with time-scaling based collision avoidance maneuvers for non-holonomic robots. 
You are also required to answer the questions mentioned in the [Assignment PDF](/Assignment3.pdf) through experiments. You should use the code written from part A of this assignment for generting non-holonomic robot trajectories. 

## Instructions:

* Submit your code files and a report in the GitHub classroom repository. Do not upload the simulation videos to the assignment repository.

* Provide the OneDrive/Google Drive links to the generated simulation videos in the report or in the repository README.

* For stitching simulation images into videos:

In `mkmovie.sh` enter the path to the directory where the snapshots are stored and edit the name of the simulation. For example, if my simulation screenshots are stored in the directory `nonhn_tree/`, I will run the following command:

`ffmpeg -r 2 -f image2 -i nonhn_tree/snap%d.png -s 1000x1000 -pix_fmt yuv420p -y simulation.mp4`


### Deadline: 

**May 1, 2022, 11:55pm**
