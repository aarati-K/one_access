# OneAccess
A unified active data storage and access layer for PyTorch

# Setup
Recommended cloudlab profile: c220g5-single-node  
`$ sudo apt update && sudo apt upgrade`  
`$ sudo apt install python python3-pip`  

Using Python 3.6+ with `virtualenv`:  
`$ sudo apt install virtualenv`  
`$ virtualenv -p $(which python3) venv`  
`$ source venv/bin/activate`  

Further steps are performed assuming the user is in virtualenv, with Python 3.6.
Installing (without CUDA):  
`$ pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl`  
`$ pip install -r requirements.txt`  

More information on installing PyTorch locally:
https://pytorch.org/get-started/locally/
