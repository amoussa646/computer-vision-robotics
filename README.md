# computer-vision-robotics

Overview
This repository integrates frequently used computer vision tasks in robotics, specifically tailored for social companion robots. These tasks enable the robot to better understand and interact with its environment, which typically includes people, objects, colors, and actions/events.


1) People 

Person Detection: 
function: Generates a bounding box surrounding a person if detected in a frame 
importance: Enables the robot to initiate interactions autonomously or execute actions such as activity tracking and surveillance.

Face Detection:
function: Generates a bounding box surrounding a the face of the person only if detected in a frame 
importance: After detecting a person, the robot locks on to the face to perform face recognition, determining if the user is known. This allows for personalized interactions.

Eyes & Smile Detection:
function: Generates a bounding box surrounding the eyes or/and the smile of a person if detected in a frame 
importance: Provides additional information about the user’s current activity and emotions, enhancing interaction.
Object Detection

2) Object Classification:
function:Generates a bounding box surrounding the known objects if detected in a frame 
importance: instead of the robot's environment being mostly continious unknown regions now there its environment is filled with recognizable stuff, making the unknown regions much smaller and not continious and can be used with the interactions with the user

3) Color Detection:
function: Generates a bounding box surrounding all the continious blobs of the selected color that are detected in a frame. 
importance: allowing the robot to recognize objects by color and shape even if the exact object classification fails. This provides a a deeper understanding of the environment, facilitating smoother interactions between the robot and users.


4) Actions/Events
Image Captioning

Describes the environment in more complex terms, such as actions or events involving people and objects. For example, it can describe a scene as “a man sitting on a chair,” helping the robot to understand the relationship between objects and actions, enhancing its mimicry of human intelligence.



Structure:

The repository contains two main folders:

Core Vision Tasks: Includes all tasks except image captioning. All combined in one server.py and can be accessed through client.py where you choose the that tast you to run and a real time feed with the results can be viewed 

Image Captioning: Dedicated to the image captioning task, with its own server.py and client.py files, also pre-configured and model-included. Just in the client side press on the button "What do you see?" and it will reply with a text expressing what it sees 



 
Future Updates
- Clear installation instructions for both folders will be provided in the upcoming days.
- The installation of the detection tasks is straight forwards but i will include the exact steps and check on the requirements.txt to avoid wasting time 
-  But for now i have to include the installation of image captioning because it tricky

  python3 -m venv imageCaption

 source imageCaption/bin/activate

 pip3 install chainer==1.19.0
 pip3 install scipy
 pip3 install h5py
 pip3 install pillow
 pip3 install fastapi
 pip3 install opencv-python-headless
 pip3 install requests

 apt-get install python-h5py (if Mac skip this line)


 then in the environment we just created go to these files:
 
 imageCaption/lib/python3.10/site-packages/chainer/link.py
imageCaption/lib/python3.10/site-packages/chainer/functions/pooling/pooling_2d.py
imageCaption/lib/python3.10/site-packages/chainer/functions/array/get_item.py
imageCaption/lib/python3.10/site-packages/chainer/functions/split_axis.py

and change every  collections.Sequence to collections.abc.Sequence
and change every  collections.Iterable to collections.abc.Iterable






