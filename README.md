# Cable-Manipulation
This repository contains the experimental data and some examples on the tracking of cable manipulation. It will start small and hopefully grow over the coming months and years as I attempt to tackle this particular problem, collate and curate literature around this area and try many differant types of solutions.

# Research Agenda:

Manufacturing is but one area that exists where the production, manipulation and assembly of may types of cables and wires are undertaken through both automated or manual processes. Naturally these processes have an impact on the final outcome of the product and its lifecycle performance, and through a better understanding of these processes, throgh capture, modelling and tracking perhaps we can provide some value, be it productivity, performance or quality outcomes. The main use case that this repository and surrounding work focuses on is the field of electrical machine (EM) manufacture, and where applicable the maniplation and assembly of cables and wires occur, for example in the coils and windings assembly process. 

A typical example can be seen here,...

# Themes and Challenges:
## Hands and end effectors
The first main theme or challenge revolves aruond the main object of interaction when it comes to the cable or wire that is being manipulated. This can be an automated robotic solution or end effector, or more commonly a hand either freely or with some tool. Our goal is to be able to record cable manipulation processes and in realtime capture useful information related to this human-machine interface.
### Hand detection
The detection of hands in a scene is probably the first task thats needs to be undertaken, ideally in a natural scene through the use of machine vision, without the need for enhancments to aid this endeavour (3D depth information, coloured gloves, sensor tracking).
### Hand segmentation  
Once we have successfully identified the hands, it would seem prudent to look into going a step further and trying to segment out the actual hand itself from the scene it is detected in. this can provide useful information it itself (sign language is one such example) but also can be used to aid further modelling or representation approaches which capture a higher fidelity model of the hand.
### Hand 3D pose estimatation 
Possibly the hardest task to undertake with regards to monitoring or tracking of hands is the transition from a 2D space to a 3D one, and the correct alignment of bones and joints in the hand. However having this information can tell us about more that is going on during manipulation, for example perhaps the path of the cable is it is held but occluded from view.
### Hand tracking
This is almost a subset of detection, in a sense once we have identified the object, i.e hand, we also need to be able to keep track of any information that might be relevent for realtime / future analysis or inference.
### Hand physical interaction
How the hand interacts with the cable directly or through the tool is also an important factor in understanding and modelling the processes that occur during manipulation. For example, what forces are applied, has there been any excessive twisting or bending, is the grip to tight, or to much pressure applied leading to insulation damage or internal breaks. the question becomes can we infer any of this through the data we capture and how we choose to possibly model it.

# Other Work and Related Material:
This section includes other work, or related material of interest to researchers or practitioners in this area.


**Pytorch Repositories:**   
https://github.com/rwightman/pytorch-image-models


# Bibliography
A list of useful literature, though not exactly exhaustive, where useful links to other repositories or groups who have collated their own lists of relevent material are included.

**Hand detection:**  
Vision-based hand pose estimation: A review
https://www.cse.unr.edu/~bebis/handposerev.pdf





























