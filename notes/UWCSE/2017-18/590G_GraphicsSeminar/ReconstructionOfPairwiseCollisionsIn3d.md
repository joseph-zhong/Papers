# Reconstruction of pairwise collisions in 3D

- Presented by Kostas Remenas 
- CSE590B Graphics Seminar Spring 2018
- SMASH: Physics-guided Reconstruction of Collisions from Videos
  - SIGGRAPH Asia 2016

## What?
- Reconstruction in 3D
- Position, Velocity, Orientation, angular velocity
- Relative mass, coefficient of restitution
- From a curtain blocking collision

## System
- Bounding Box for the objects
- Extract frames from video 
 - Label 3D orientation
 - Labelled 2D positions with initial parabolas
- Physics model is used as a regularizer


Ransac Labelling
- Assume mid frame as collision
- Fit parabola 

## Optimization



