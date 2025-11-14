python -m src.traj_generation.trajectory_viz

Use 9P model to model trajectory visualizer, and then work on moving it into a "virtual camear" 


Couple more TODOs: 
- Fix up the old trajectory, prediction 
- Add in spin axis for the visualization 
- Collect a bunch of data. Format it for model
- See what models it makes sense to run on synthetic data
- Add in distortions + motion blurr to make it more realistic (radial distortion, vignetting, rolling shutter)
- Motion blurr,



Model: 
- Object detection --> 2D bounding box --> (size known) trajectory (3D) --> Prediction on where its going to land